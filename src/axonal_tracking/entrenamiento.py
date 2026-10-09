"""Entrenamiento de KymoRoPE sobre el `Trainer` de `notebooks/trainer.py`.

En vez de un bucle propio, se reusa `Trainer.train_model_v2` (AMP, acumulacion de
gradiente, clipping, checkpointing, scheduler en el borde de acumulacion) y se
sobreescribe **solo** el punto donde un batch se convierte en perdida. Ese es el
unico lugar donde KymoRoPE no encaja con la clase base:

  `Trainer` asume            KymoRoPE entrega
  ------------------------   ----------------------------------------------------
  `(input, target)` densos   `LoteEmpaquetado` -- lista de muestras de tamano
                             distinto, sin apilar (esa es toda la gracia: no se
                             redimensiona nada, `plan/kymorope-guide.md` SS2.3)
  `model(x) -> tensor`       `model(lista) -> lista de SalidaKymoRoPE` (3 heads)
  `loss_fn(out, target)`     `perdida_total(salida, muestra) -> dict con desglose`

**Ubicacion**: `Trainer` vive en `notebooks/trainer.py`, asi que este modulo lo
importa agregando `notebooks/` al path (mismo patron que
`scripts/entrenar_atencion_v1.py` con `scripts/`). Mover `trainer.py` a
`src/axonal_tracking/` sacaria esa verruga, pero es un archivo de autoria ajena y
no se toca sin pedirlo.

**Parada temprana**: `Trainer.eval_model()` devuelve perdida de PIXEL. Eso NO es un
criterio valido de parada para este problema -- `plan/kymorope-guide.md` SS7 y
`plan/decnet-finetune-guide.md` piden `frac_id_switch` a nivel TRAYECTORIA, y las
dos metricas no se mueven juntas. Hasta que exista la decodificacion de SS2.6,
`metrica_parada_temprana()` levanta `NotImplementedError` a proposito: es mejor que
falle fuerte a que un `EarlyStopping` mal cableado parezca funcionar.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from axonal_tracking.datos_pixel import (
    MODO_NORMALIZACION_DEFAULT,
    DatasetKymografos,
    LoteEmpaquetado,
    agrupar_por_tokens,
    collate_empaquetado,
    modo_de_checkpoint,
)
from axonal_tracking.kymorope import PARCHE, KymoRoPE, memoria_total, perdida_total

_RAIZ = Path(__file__).resolve().parents[2]
if str(_RAIZ / "notebooks") not in sys.path:
    sys.path.append(str(_RAIZ / "notebooks"))

from trainer import EarlyStopping, Trainer

__all__ = [
    "EarlyStopping",
    "MuestreadorPorTokens",
    "TrainerKymoRoPE",
    "construir_entrenador",
    "metrica_parada_temprana",
]

# Tope de tokens por lote. **Va atado a `kymorope.PARCHE`**: el presupuesto esta en
# TOKENS pero el costo del stem y del decoder esta en PIXELES, y un token cubre
# `PARCHE^2` px. Con PARCHE=32, 2048 tokens ~= 2.1 Mpx, el mismo presupuesto de
# pixeles que daban 8192 tokens con PARCHE=16. Si se sube el parche sin bajar esto,
# entran ~2.7x mas pixeles por lote y la memoria SUBE en vez de bajar (medido:
# 10.31 GB con parche 32 y 8192 tokens, contra 2.45 GB con 2048).
MAX_TOKENS_LOTE = 2048
MAX_MUESTRAS_LOTE = 8


class MuestreadorPorTokens(Sampler):
    """`batch_sampler` que agrupa indices por CONTEO DE TOKENS (SS2.3).

    Re-agrupa en cada `__iter__` con una semilla distinta, llevando el contador de
    epoca adentro: un `set_epoch()` externo se olvida, un contador interno no. Todo es
    determinista dado `(semilla, epoca)`, asi que el plan completo de lotes -- y con el
    la cantidad de pasos del scheduler -- se puede calcular de antemano
    (`pasos_totales`), y reanudar en la epoca `e` reproduce exactamente la epoca `e`.

    `muestras_por_epoca`: con 30 000 muestras una epoca completa son ~45 min en la 3060.
    Si se fija, cada "epoca" toma el siguiente tramo de ese tamano de una permutacion del
    dataset (una permutacion nueva por vuelta completa, todas con semilla), asi que cada
    muestra se ve una vez por vuelta pero hay checkpoint y validacion cada tramo. Sin
    fijarlo (default) el comportamiento es el de siempre: todas las muestras por epoca,
    mismo orden de lotes que antes de existir el parametro."""

    def __init__(
        self,
        formas: list[tuple[int, int]],
        *,
        patch: int = PARCHE,
        max_tokens: int = MAX_TOKENS_LOTE,
        max_muestras: int = MAX_MUESTRAS_LOTE,
        semilla: int = 0,
        muestras_por_epoca: int | None = None,
    ):
        self.formas = formas
        self.patch = patch
        self.max_tokens = max_tokens
        self.max_muestras = max_muestras
        self.semilla = semilla
        n = len(formas)
        self.muestras_por_epoca = (
            muestras_por_epoca if muestras_por_epoca and muestras_por_epoca < n else None
        )
        self._epoca = 0
        self._lotes_cache: dict[int, list[list[int]]] = {}
        self._permutaciones: dict[int, np.ndarray] = {}

    def _indices_epoca(self, epoca: int) -> np.ndarray:
        n = len(self.formas)
        if self.muestras_por_epoca is None:
            return np.arange(n)
        k = self.muestras_por_epoca
        ini = epoca * k
        partes, pos = [], ini
        while pos < ini + k:  # el tramo puede cruzar de una vuelta a la siguiente
            vuelta, desde = divmod(pos, n)
            if vuelta not in self._permutaciones:
                gen = np.random.default_rng(self.semilla + 1_000_003 * (vuelta + 1))
                self._permutaciones[vuelta] = gen.permutation(n)
            toma = min(ini + k - pos, n - desde)
            partes.append(self._permutaciones[vuelta][desde:desde + toma])
            pos += toma
        return np.concatenate(partes)

    def _calcular_lotes(self, epoca: int) -> list[list[int]]:
        idx = self._indices_epoca(epoca)
        lotes = agrupar_por_tokens(
            [self.formas[i] for i in idx],
            patch=self.patch,
            max_tokens=self.max_tokens,
            max_muestras=self.max_muestras,
            barajar=True,
            semilla=self.semilla + epoca,
        )
        return [[int(idx[j]) for j in lote] for lote in lotes]

    def lotes_de_epoca(self, epoca: int) -> list[list[int]]:
        """Lotes (indices del dataset) de la epoca `epoca`, deterministas."""
        if epoca not in self._lotes_cache:
            self._lotes_cache[epoca] = self._calcular_lotes(epoca)
        return self._lotes_cache[epoca]

    def pasos_totales(self, epocas: int, gradient_accumulation_steps: int) -> int:
        """Pasos de optimizador de `epocas` epocas, exactos: `train_model_v2` da un paso
        cada `gradient_accumulation_steps` lotes mas uno final si sobran. Calcula sin
        guardar los planes (30k muestras x 40 epocas serian ~40 MB de listas)."""
        return sum(
            -(-len(self._calcular_lotes(e)) // gradient_accumulation_steps) for e in range(epocas)
        )

    def __iter__(self):
        lotes = self.lotes_de_epoca(self._epoca)
        self._lotes_cache.pop(self._epoca - 1, None)  # no acumular planes viejos
        self._epoca += 1
        yield from lotes

    def __len__(self) -> int:
        return len(self.lotes_de_epoca(self._epoca))


class TrainerKymoRoPE(Trainer):
    """`Trainer` con `compute_loss` / `compute_eval_loss` adaptados a KymoRoPE.

    `loss_fn` queda en `None`: la perdida de este modelo es
    `kymorope.perdida_total(salida, muestra)`, que toma la muestra entera (tres
    targets + dos mascaras de supervision) y devuelve el desglose, no un par
    `(output, target)`. El resto de `train_model_v2` se usa sin tocar."""

    def __init__(
        self,
        model: KymoRoPE,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device,
        *,
        gradient_accumulation_steps: int = 1,
        lambdas: dict[str, float] | None = None,
        save_dir: str | Path = "results/kymorope/checkpoints",
        save_every_n: int = 1000,
    ):
        super().__init__(
            model=model,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            loss_fn=None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=str(save_dir),
            save_every_n=save_every_n,
        )
        self.lambdas = lambdas or {}
        self.desglose: dict[str, list[float]] = defaultdict(list)
        # Los fija `construir_entrenador`: el muestreador (cuantas epocas van completas) y
        # los argumentos que definen el plan de pasos del scheduler (ver `cargar_checkpoint`)
        self.muestreador: MuestreadorPorTokens | None = None
        self.plan: dict = {}

    # -- utilidades ------------------------------------------------------- #
    def _mover(self, lote: LoteEmpaquetado) -> list:
        """Los tensores de cada muestra al dispositivo. `LoteEmpaquetado.a` no se
        usa para no depender de que el collate haya devuelto ese tipo exacto (con
        `num_workers>0` el objeto se reconstruye del otro lado del pickle)."""
        campos = ("kymo", "trackness", "instancias", "theta", "mask_pull", "mask_theta")
        muestras = lote.muestras if hasattr(lote, "muestras") else lote
        return [
            replace(m, **{c: getattr(m, c).to(self.device) for c in campos}) for m in muestras
        ]

    def _perdida_lote(self, muestras: list, use_amp: bool, dtype: torch.dtype):
        """Media POR MUESTRA del desglose. Media y no suma: una muestra ancha tiene
        mas pixeles pero no vale mas que una angosta.

        El autocast envuelve solo el forward del modelo; las perdidas se calculan en
        float32. La discriminativa hace normas y distancias entre medias de cluster
        y en float16 eso subdesborda -- es el patron estandar de AMP (forward en baja
        precision, perdida en alta).

        `cache_enabled=False` cuando el modelo usa gradient checkpointing: el cache
        de pesos casteados de autocast se llena en el forward original y ya esta
        caliente en el recomputo, asi que las dos pasadas guardan distinta cantidad
        de tensores y `torch.utils.checkpoint` aborta. Es la combinacion que la doc
        de PyTorch pide desarmar explicitamente."""
        usa_ckpt = bool(getattr(self.model, "usar_checkpoint", False))
        with torch.autocast(
            device_type=self._device_type(),
            dtype=dtype,
            enabled=use_amp,
            cache_enabled=not usa_ckpt,
        ):
            salidas = self.model(muestras)
        salidas = [
            replace(
                s,
                trackness=s.trackness.float(),
                embedding=s.embedding.float(),
                orientacion=s.orientacion.float(),
            )
            for s in salidas
        ]
        desgloses = [perdida_total(s, m, **self.lambdas) for s, m in zip(salidas, muestras)]
        return {
            clave: torch.stack([d[clave] for d in desgloses]).mean()
            for clave in ("total", "trackness", "embedding", "orientacion")
        }

    # -- puntos de extension de `Trainer` --------------------------------- #
    def compute_loss(self, batch, use_amp: bool, dtype: torch.dtype):
        partes = self._perdida_lote(self._mover(batch), use_amp, dtype)
        for clave, valor in partes.items():
            if clave != "total":
                self.desglose[clave].append(float(valor.detach()))
        return partes["total"]

    def compute_eval_loss(self, batch):
        return self._perdida_lote(self._mover(batch), use_amp=False, dtype=torch.float32)["total"]

    def save_checkpoint(self, step: int, final: bool = False):
        """`Trainer.save_checkpoint` + lo que hace falta para reanudar.

        `Trainer` guarda solo `step`, que `train_model_v2` reinicia en cada epoca: el
        checkpoint de la corrida de 40 epocas del 2026-09-24 dice `step=153` (lotes de la
        ULTIMA epoca) con el scheduler en 3080/3080. Reanudar desde ahi no sabia en que
        epoca estaba y rearmaba 40 epocas encima de un OneCycle terminado, que levanta
        error en el primer paso. Aca se guardan ademas `epocas_completas` (se llama al
        final de `train_model_v2`, con la epoca ya entrenada) y el `plan`."""
        os.makedirs(self.save_dir, exist_ok=True)
        sufijo = "final" if final else f"step_{step}"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": step,
                "epocas_completas": self.muestreador._epoca if self.muestreador else None,
                "plan": self.plan,
            },
            os.path.join(self.save_dir, f"checkpoint_{sufijo}.pt"),
        )

    def cargar_checkpoint(self, ruta: str | Path, *, estricto: bool = True) -> int:
        """Restaura modelo, optimizador, scheduler, scaler y la epoca del muestreador.
        Devuelve cuantas epocas completas trae el checkpoint.

        Solo reanuda una corrida con el MISMO plan (epocas, muestras por epoca, tope de
        tokens, acumulacion, semilla, cantidad de muestras, lr): el estado del OneCycle
        que se carga fue dimensionado para ese plan, y con otro quedaria desfasado o se
        pasaria de su total. Un checkpoint anterior a este cambio no trae
        `epocas_completas` y no se puede reanudar: para arrancar de sus pesos esta
        `construir_entrenador(pesos_iniciales=...)`, que es un fine-tune.

        `estricto=False` tolera que no calce el estado del optimizador."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el checkpoint {ruta}")
        # `weights_only=False`: el checkpoint trae estados de optimizador/scheduler,
        # no solo tensores. Es un archivo propio, no de terceros.
        ck = torch.load(ruta, map_location=self.device, weights_only=False)
        if ck.get("epocas_completas") is None:
            raise ValueError(
                f"{ruta} no guarda `epocas_completas` (checkpoint anterior al 2026-09-27): "
                "reanudarlo desfasaria el scheduler. Para arrancar de sus pesos, usar "
                "`construir_entrenador(pesos_iniciales=...)`."
            )
        # los planes anteriores al 2026-10-08 no traen el modo: eran todos "p50_clip"
        plan_ck = {"modo_normalizacion": modo_de_checkpoint(ck), **(ck.get("plan") or {})}
        if ck.get("plan") and self.plan and plan_ck != self.plan:
            raise ValueError(
                f"El checkpoint es de otro plan de entrenamiento:\n  checkpoint {plan_ck}\n"
                f"  esta corrida {self.plan}\nReanudar con los mismos argumentos, o usar "
                "`pesos_iniciales` para un fine-tune nuevo desde esos pesos."
            )
        self.model.load_state_dict(ck["model_state_dict"])
        if ck.get("optimizer_state_dict") is not None:
            try:
                self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            except ValueError:
                if estricto:
                    raise
        if ck.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        if ck.get("scaler_state_dict") is not None and self.scaler is not None:
            self.scaler.load_state_dict(ck["scaler_state_dict"])
        epocas = int(ck["epocas_completas"])
        if self.muestreador is not None:
            self.muestreador._epoca = epocas  # la proxima epoca reproduce el plan original
        return epocas

    def desglose_medio(self) -> dict[str, float]:
        """Media del desglose acumulado y reinicio del acumulador. Se mira por
        separado a proposito: si el embedding colapsa mientras el trackness baja, el
        total no lo muestra."""
        medias = {k: sum(v) / len(v) for k, v in self.desglose.items() if v}
        self.desglose.clear()
        return medias


def metrica_parada_temprana(*_args, **_kwargs) -> float:
    """Metrica de parada temprana a nivel TRAYECTORIA. Todavia no existe.

    Requiere la decodificacion de `plan/kymorope-guide.md` SS2.6 (umbral de
    trackness -> clustering de embeddings -> `evaluacion.extraer_subpixel` ->
    polilineas) y despues `evaluacion.evaluar_trayectorias_polilineas`.

    Levanta en vez de caer a la perdida de pixel a proposito: SS7 de la guia y
    `plan/decnet-finetune-guide.md` piden parar por `frac_id_switch`, y usar la
    perdida de pixel en su lugar daria una corrida que parece bien parada y no lo
    esta."""
    raise NotImplementedError(
        "Falta la decodificacion a trayectorias (kymorope-guide SS2.6). "
        "No usar `eval_model()` (perdida de pixel) como criterio de parada."
    )


def construir_entrenador(
    raiz_datasets: Path | str,
    *,
    epocas: int = 40,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    gradient_accumulation_steps: int = 2,
    max_tokens: int = MAX_TOKENS_LOTE,
    fps_excluido: float | None = None,
    limite_train: int | None = None,
    subconjunto_train: int | None = None,
    limite_val: int | None = None,
    num_workers: int | None = None,
    contexto_mp: str | None = None,
    cache_dir: Path | str | None = None,
    dispositivo=None,
    save_dir: Path | str | None = None,
    usar_checkpoint: bool | None = None,
    reanudar: bool | str | Path = False,
    pesos_iniciales: str | Path | None = None,
    muestras_por_epoca: int | None = None,
    procesos_cache: int = 1,
    semilla: int = 0,
    modo_normalizacion: str = MODO_NORMALIZACION_DEFAULT,
) -> tuple[TrainerKymoRoPE, dict]:
    """Arma datasets, samplers, modelo, optimizador, scheduler y el `Trainer`.

    `fps_excluido` implementa el ablation de SS5.3: saca una tasa de muestreo del
    train y la deja en val/test. Es un filtro sobre el `Dataset`, sin regenerar nada
    -- las cinco tasas estan en los tres splits.

    **Tres formas de arrancar** (y `info["epoca_inicial"]` dice desde que epoca seguir):

    - de cero (default);
    - `pesos_iniciales=ruta.pt`: **fine-tune**. Carga SOLO los pesos del modelo de otra
      corrida; optimizador y scheduler arrancan nuevos, dimensionados para esta. Es lo
      que corresponde para seguir entrenando el checkpoint del 2026-09-24 sobre el
      dataset de 30k (con un `lr` mas bajo que el de un entrenamiento de cero);
    - `reanudar=True` (o una ruta): **continuar una corrida cortada** con los mismos
      argumentos. Restaura todo, incluida la epoca del muestreador, y se niega si el
      checkpoint es de otro plan (`TrainerKymoRoPE.cargar_checkpoint`).

    `subconjunto_train=N` entrena sobre N muestras representativas y anidadas del train
    (`DatasetKymografos._subconjunto_anidado`: siempre incluyen las 800 originales, y el
    de 5000 esta contenido en el de 10000). Es para la curva de escala de datos; NO usar
    `limite_train` para eso, que toma las primeras N y en el dataset de 33k son casi todas
    de un solo perfil.

    `modo_normalizacion` es la normalizacion de entrada (`datos_pixel.MODOS_NORMALIZACION`,
    `plan/kymorope-preprocessing.md`). Queda en el `plan` del checkpoint y la inferencia la
    lee de ahi (`datos_pixel.modo_de_checkpoint`). Un fine-tune desde pesos entrenados con
    otro modo es valido (los pesos se adaptan a la entrada nueva) y se avisa; una
    reanudacion con otro modo, no.

    `muestras_por_epoca`: epocas cortas sobre datasets grandes (ver
    `MuestreadorPorTokens`); con 30k muestras, p.ej. 5000 da checkpoint y validacion cada
    ~7 min en vez de cada ~45. Es otra cosa que `subconjunto_train`: recorre TODO el train,
    de a tramos. `procesos_cache` paraleliza el precalculo del cache si
    falta (en Windows desde un notebook funciona porque la tarea vive en un modulo;
    para 30k conviene correr antes `scripts/precalcular_cache_kymorope.py`).

    **Parche del stem** (`kymorope.PARCHE`, hoy 32): con el presupuesto de pixeles
    igualado, el parche 32 gana en las dos cosas contra el 16 -- 2.45 GB contra 3.78 y
    0.83 s/paso contra 2.59, o sea ~2.1 min/epoca contra 6.4. La atencion va como
    `B*N^2` y el parche 32 divide `N` por 4. El precio es contexto mas grueso (cada
    token cubre 32x32 px en vez de 16x16); la resolucion de SALIDA no cambia porque el
    decoder sigue recibiendo skips en s2/s4. **Ese precio no esta medido en calidad**:
    solo lo decide el gate de `plan/kymorope-guide.md` SS7. `KymoRoPE(parche=16)`
    vuelve al anterior.

    `usar_checkpoint=None` lo activa en MPS, y **no es opcional a este tamano**:
    `scaled_dot_product_attention` en MPS MATERIALIZA la matriz de atencion completa
    (no hay kernel fusionado, con o sin mascara). Medido a `B=3, H=6, N=2726`: 0.57 GB
    por capa x 12 capas = ~6.8 GB, que es el grueso del consumo. Sobre el peor lote
    real del dataset:

        max_tokens  checkpoint  activaciones  s/paso  min/epoca
        8192        False          12.60 GB     2.07      5.1
        8192        True            3.78 GB     2.38      5.9   <- default
        4096        False           9.64 GB     1.54      7.7
        4096        True            2.36 GB     1.50      7.5

    O sea: 3.3x menos memoria por 15% de tiempo. Bajar `max_tokens` ademas acota
    `N_max`, pero multiplica la cantidad de pasos y sale mas caro por epoca.

    **Cuidado**: con checkpointing, interrumpir una celda en pleno `backward()` deja
    instalado el hook de tensores guardados y todo forward posterior en ese kernel
    aborta con `CheckpointError`. La unica salida es reiniciar el kernel.
    """
    from axonal_tracking.kymorope import dispositivo_preferido

    dev = dispositivo or dispositivo_preferido()
    # Cache de targets en disco. Construirlos cuesta ~63 ms de CPU por muestra
    # (rasterizar + dilatar ~30 trazas) y son DETERMINISTAS, asi que rehacerlos en
    # cada epoca es trabajo puro al pedo: 800 muestras x 40 epocas = ~34 min de CPU.
    # Cacheados, `__getitem__` baja a ~0.7 ms (86x) y el cuello de botella deja de
    # ser la carga de datos. Ocupa ~3.4 GB por split, en `results/` (gitignoreado).
    cache_raiz = Path(cache_dir) if cache_dir else _RAIZ / "results" / "kymorope" / "cache"

    if num_workers is None:
        # Con el cache caliente los workers no hacen falta: 0.7 ms de CPU por muestra
        # contra ~130 ms de GPU por lote. Y sin workers no hay procesos extra que
        # multipliquen la memoria ni la fragilidad de fork/spawn en un kernel de
        # Jupyter. Se puede subir a mano si el cache esta frio.
        num_workers = 0
    if usar_checkpoint is None:
        # Decidir por MEMORIA, no por el nombre del backend. En MPS siempre: SDPA
        # materializa la matriz de atencion completa (no hay kernel fusionado) y eso
        # domina el consumo. En CUDA hay FlashAttention, asi que solo hace falta con
        # VRAM chica -- un 3060 de 6 GB si, un T4/A100 no.
        tipo = torch.device(dev).type
        usar_checkpoint = tipo == "mps" or (tipo == "cuda" and memoria_total(dev) < 12e9)
    raiz = Path(raiz_datasets)

    fps_train = None
    if fps_excluido is not None:
        fps_train = set(DatasetKymografos(raiz / "train").fps()) - {fps_excluido}

    train = DatasetKymografos(
        raiz / "train", fps_permitidos=fps_train, limite=limite_train,
        subconjunto=subconjunto_train, cache_dir=cache_raiz / "train",
        modo_normalizacion=modo_normalizacion,
    )
    val = DatasetKymografos(
        raiz / "val", limite=limite_val, cache_dir=cache_raiz / "val",
        modo_normalizacion=modo_normalizacion,
    )
    for nombre, d in (("train", train), ("val", val)):
        faltan = sum(1 for i in range(len(d)) if not d._ruta_cache(i).exists())
        if faltan:
            print(f"precalculando cache de {nombre}: {faltan} muestras...")
            d.precalcular(verboso=False, procesos=procesos_cache)

    sampler_train = MuestreadorPorTokens(
        train.formas(), max_tokens=max_tokens, semilla=semilla,
        muestras_por_epoca=muestras_por_epoca,
    )
    sampler_val = MuestreadorPorTokens(val.formas(), max_tokens=max_tokens)
    # `num_workers>0` es lo que despega la GPU: construir los targets de una muestra
    # cuesta ~0.064 s de CPU (rasterizar + dilatar ~30 trazas), o sea ~0.45 s por
    # lote de 7. Con 0 workers eso corre SINCRONICO entre pasos y la GPU se queda
    # esperando -- el sintoma es uso de GPU ~35% con CPU al tope. Con workers se
    # solapa con el paso anterior. `persistent_workers` evita respawnearlos en cada
    # epoca (en macOS el metodo de arranque es spawn y eso cuesta).
    # Nada de `pin_memory`: en memoria unificada no aporta.
    extra: dict = {}
    if num_workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 2}
        # Se deja el arranque POR DEFECTO de la plataforma (spawn en macOS).
        # `fork` se probo y salia ~6% mas barata (1.69 vs 1.80 s/lote) porque no
        # re-importa `__main__`, pero forkear un proceso que ya tiene hilos
        # (numpy/BLAS, el runtime de MPS) puede dejar al hijo trabado con un lock
        # tomado. Es NO determinista: aca funciono varias veces seguidas y despues
        # colgo el kernel entero con los workers vivos. 6% no paga eso.
        # `contexto_mp="fork"` sigue disponible para quien lo quiera medir.
        if contexto_mp:
            extra["multiprocessing_context"] = contexto_mp
    dl_train = DataLoader(
        train,
        batch_sampler=sampler_train,
        collate_fn=collate_empaquetado,
        num_workers=num_workers,
        **extra,
    )
    dl_val = DataLoader(
        val,
        batch_sampler=sampler_val,
        collate_fn=collate_empaquetado,
        num_workers=num_workers,
        **extra,
    )

    modelo = KymoRoPE(usar_checkpoint=usar_checkpoint).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    # el scheduler avanza una vez por PASO DE OPTIMIZADOR, no por micro-batch. El total
    # sale del plan exacto del muestreador: con `muestras_por_epoca` cada epoca tiene una
    # cantidad de lotes distinta, y un OneCycle que se queda corto levanta error.
    pasos_por_epoca = -(-len(sampler_train) // gradient_accumulation_steps)  # la primera
    total_steps = max(sampler_train.pasos_totales(epocas, gradient_accumulation_steps), 1)
    # `OneCycleLR` divide por el largo de cada fase: con `pct_start*total_steps < 2`
    # la fase de calentamiento queda de largo 0 y `get_lr()` levanta
    # ZeroDivisionError. Pasa facil en corridas cortas (`limite_train`, smoke tests),
    # y ademas un ciclo completo sobre unos pocos pasos no significa nada -- ahi LR
    # constante es lo honesto, y queda dicho en `info` para que no sorprenda.
    if total_steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total_steps, pct_start=0.1
        )
        nombre_sched = "OneCycleLR"
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        nombre_sched = f"constante (corrida corta: {total_steps} pasos < 20)"

    entrenador = TrainerKymoRoPE(
        modelo,
        dl_train,
        dl_val,
        optimizer=opt,
        scheduler=sched,
        device=dev,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_dir=save_dir or (_RAIZ / "results" / "kymorope" / "checkpoints"),
    )
    entrenador.muestreador = sampler_train
    entrenador.plan = {
        "epocas": epocas, "muestras_por_epoca": sampler_train.muestras_por_epoca,
        "max_tokens": max_tokens, "gradient_accumulation_steps": gradient_accumulation_steps,
        "semilla": semilla, "n_train": len(train), "lr": lr,
        "subconjunto_train": subconjunto_train, "limite_train": limite_train,
        "modo_normalizacion": modo_normalizacion,
    }

    epoca_inicial = 0
    if reanudar:
        base = Path(save_dir or (_RAIZ / "results" / "kymorope" / "checkpoints"))
        ruta = base / "checkpoint_final.pt" if reanudar is True else Path(reanudar)
        if ruta.exists():
            epoca_inicial = entrenador.cargar_checkpoint(ruta)
            print(f"reanudado desde {ruta}: {epoca_inicial} de {epocas} epocas completas")
        else:
            print(f"reanudar activo pero no hay checkpoint en {ruta}: se arranca de cero")
    if pesos_iniciales and epoca_inicial == 0:
        ck = torch.load(pesos_iniciales, map_location=dev, weights_only=False)
        modelo.load_state_dict(ck["model_state_dict"])
        print(f"fine-tune: pesos iniciales de {pesos_iniciales} (optimizador y scheduler nuevos)")
        if modo_de_checkpoint(ck) != modo_normalizacion:
            print(f"  esos pesos se entrenaron con normalizacion '{modo_de_checkpoint(ck)}'; "
                  f"esta corrida usa '{modo_normalizacion}'")

    info = {
        "dispositivo": str(dev),
        "epoca_inicial": epoca_inicial,
        "pesos_iniciales": str(pesos_iniciales) if pesos_iniciales and epoca_inicial == 0 else None,
        "n_train": len(train),
        "subconjunto_train": subconjunto_train,
        "n_val": len(val),
        # leer el config.yaml de cada muestra solo para informar: con 30k son ~1 min
        "fps_train": sorted(set(train.fps())) if len(train) <= 5000 else "(no leido: >5000 muestras)",
        "fps_excluido": fps_excluido,
        "modo_normalizacion": modo_normalizacion,
        "muestras_por_epoca": sampler_train.muestras_por_epoca or len(train),
        "lotes_por_epoca": len(sampler_train),
        "pasos_optimizador_por_epoca": pasos_por_epoca,
        "total_steps_scheduler": total_steps,
        "scheduler": nombre_sched,
        "usar_checkpoint": modelo.usar_checkpoint,
        "num_workers": num_workers,
        "cache_dir": str(train.cache_dir),
        "parametros": sum(p.numel() for p in modelo.parameters()),
    }
    return entrenador, info
