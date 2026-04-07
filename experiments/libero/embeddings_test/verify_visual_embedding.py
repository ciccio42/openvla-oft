"""
verify_visual_embedding.py

Verifica se l'embedding visivo (DINOv2+SigLIP → vision_backbone → projector)
è simile tra i 10 task di LIBERO Goal.

Se i projected_patches sono quasi identici tra task diversi, significa che
il modello deve affidarsi quasi esclusivamente al prompt testuale per
distinguere quale azione eseguire. Di conseguenza, le variazioni di
performance su L1/L2/L3 sono attribuibili alla componente linguistica.

Output:
  - Matrice 10x10 di cosine similarity tra visual embedding dei task
  - Matrice 10x10 di euclidean distance
  - Sanity check: stesso frame, prompt diversi → projected_patches identici?

--- SCOPO GENERALE ---
Questo script risponde alla domanda: "I visual embedding sono sufficientemente
diversi tra i 10 task LIBERO Goal da contribuire alla discriminazione dei task,
oppure il modello si affida SOLO al testo?"

Pipeline:
1. Carica il modello OpenVLA-OFT (checkpoint 20000).
2. Per ognuno dei 10 task, inizializza la scena LIBERO e cattura il primo frame.
3. Passa ogni frame attraverso il blocco visivo (vision_backbone + projector)
   e calcola il mean-pool dei patch embedding proiettati.
4. Costruisce una matrice 10x10 di cosine similarity e euclidean distance
   tra i visual embedding di tutti i task.
5. Esegue un sanity check: verifica che, dato lo STESSO frame con prompt
   DIVERSI, i projected_patches siano identici (conferma che il blocco
   visivo è indipendente dal testo quando use_film=False).
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# ─────────────────── path setup ───────────────────
# Risolve il percorso assoluto di QUESTO file (elimina symlink)
current_file = Path(__file__).resolve()
# Risale 3 livelli nella gerarchia: libero/ → experiments/ → openvla-oft/ (root progetto)
project_root = current_file.parent.parent.parent        # openvla-oft/
# Dalla parent di openvla-oft (robosuite_test/) entra in LIBERO/
libero_root  = project_root.parent / "LIBERO"
# Inserisce le root in cima al sys.path per consentire gli import dal progetto
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(libero_root))

# --- Import dalle utility del progetto OpenVLA-OFT ---
# get_processor: carica tokenizer + image preprocessor. Input: cfg → Output: callable(prompt, img) → dict tensori
# get_vla: carica il modello VLA completo dal checkpoint. Input: cfg → Output: modello con vision_backbone, projector, language_model
# get_action_head: carica la testa di azione MLP. Input: cfg, llm_dim → Output: nn.Module
# get_proprio_projector: carica il proiettore propriocettivo. Input: cfg, llm_dim, proprio_dim → Output: nn.Module
# resize_image_for_policy: ridimensiona un'immagine numpy. Input: (ndarray, int) → Output: ndarray ridimensionato
from experiments.openvla_utils import (
    get_processor,
    get_vla,
    get_action_head,
    get_proprio_projector,
    resize_image_for_policy,
)
# get_libero_env: crea ambiente simulazione LIBERO. Input: (task, bool, int) → Output: (env, desc, bddl_path)
# get_libero_image: estrae immagine frontale da obs. Input: dict → Output: ndarray (H,W,3)
# get_libero_wrist_image: estrae immagine wrist da obs. Input: dict → Output: ndarray (H,W,3)
# get_libero_dummy_action: genera azione nulla (zero). Input: str → Output: ndarray
from experiments.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    get_libero_dummy_action,
)
# get_image_resize_size: calcola dimensione resize per la policy. Input: cfg → Output: int
from experiments.robot_utils import get_image_resize_size
# benchmark: modulo LIBERO con definizioni delle task suite.
# benchmark.get_benchmark_dict() → dict {nome_suite: classe_suite}
from libero.libero import benchmark

# ─────────────────── constants ───────────────────

# Percorso assoluto al checkpoint 20000 del modello OpenVLA-OFT fine-tuned su LIBERO Goal.
# Contiene pesi del VLM (vision backbone + projector + LLM) + action_head + proprio_projector.
CHECKPOINT_PATH = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/openvla/"
    "openvla-7b+libero_goal_no_noops_20000_chkpt"
)
# Seleziona il device di computazione: GPU 0 se disponibile, altrimenti CPU
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Lista ordinata dei 10 task della suite LIBERO Goal (indice = task_id).
# L'ordine corrisponde a quello definito in tasks_info.txt del benchmark.
TASK_NAMES = [
    "put_the_wine_bottle_on_top_of_the_cabinet",     # task 0
    "open_the_top_drawer_and_put_the_bowl_inside",   # task 1
    "turn_on_the_stove",                             # task 2
    "put_the_bowl_on_top_of_the_cabinet",            # task 3
    "put_the_bowl_on_the_plate",                     # task 4
    "put_the_wine_bottle_on_the_rack",               # task 5
    "put_the_cream_cheese_in_the_bowl",              # task 6
    "open_the_middle_drawer_of_the_cabinet",         # task 7
    "push_the_plate_to_the_front_of_the_stove",      # task 8
    "put_the_bowl_on_the_stove",                     # task 9
]

# Prompt diversi usati nel SANITY CHECK per verificare che il blocco visivo
# (vision_backbone + projector) sia indipendente dal testo.
# Se use_film=False, cambiare il prompt NON deve modificare i projected_patches.
# Usiamo 4 prompt semanticamente diversi applicati alla STESSA immagine.
SANITY_PROMPTS = [
    "In: What action should the robot take to put the wine bottle on the top of the drawer?\nOut:",
    "In: What action should the robot take to open the middle layer of the drawer?\nOut:",
    "In: What action should the robot take to put the bowl on the stove?\nOut:",
    "In: What action should the robot take to place the wine bottle on the top of the drawer?\nOut:",
]


# ─────────────────── config ───────────────────
# Dataclass di configurazione con tutti i parametri per il caricamento del modello
# e l'estrazione degli embedding visivi. Identica a quella di compare_embeddings.
@dataclass
class Cfg:
    # Percorso al checkpoint pre-addestrato/fine-tuned
    pretrained_checkpoint: str = CHECKPOINT_PATH
    # Famiglia del modello: "openvla" determina formato I/O
    model_family: str = "openvla"
    # Se True, carica la testa di azione con regressione L1
    use_l1_regression: bool = True
    # Se True, usa diffusion per predire azioni (alternativa a L1)
    use_diffusion: bool = False
    # Passi di denoising per la diffusione
    num_diffusion_steps: int = 50
    # Se True, usa FiLM conditioning: il testo modula le feature visive.
    # CRUCIALE per questo script: se False, il blocco visivo è indipendente dal testo.
    use_film: bool = False
    # Numero immagini in input: 2 = agentview + wrist camera
    num_images_in_input: int = 2
    # Se True, include dati propriocettivi del robot
    use_proprio: bool = True
    # Se True, applica center crop alle immagini
    center_crop: bool = True
    # Numero di azioni predette per forward pass (open-loop chunking)
    num_open_loop_steps: int = 8
    # Quantizzazione 8-bit o 4-bit (riduce memoria, non usata qui)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # Nome della task suite LIBERO
    task_suite_name: str = "libero_goal"
    # Chiave normalizzazione azioni (impostata in load_model)
    unnorm_key: str = ""
    # Risoluzione immagini dell'ambiente (pixel per lato)
    env_img_res: int = 256
    # Passi di stabilizzazione con azioni nulle prima di catturare il frame
    num_steps_wait: int = 10


# ─────────────────── model loading ───────────────────
def load_model(cfg: Cfg):
    """
    Carica il modello OpenVLA-OFT dal checkpoint.

    A differenza di compare_embeddings, qui NON restituisce action_head e proprio_projector
    come output (li carica solo per verificarne l'esistenza, poi li scarta).
    Questo script ha bisogno solo di: model (per vision_backbone + projector) e processor.

    Input:
        cfg: Cfg — configurazione con tutti i parametri.

    Output (tuple):
        model       — modello VLA completo con .vision_backbone, .projector, .language_model
        processor   — callable: processor(prompt, image) → dict con input_ids, attention_mask, pixel_values
        resize_size — int: dimensione lato per il resize delle immagini (es. 224)
    """
    print(f"Loading model from:\n  {cfg.pretrained_checkpoint}\n")
    # Carica il modello VLA dal checkpoint: include ViT, projector MLP, LLaMA 7B
    model = get_vla(cfg)

    # Determina la chiave per le statistiche di normalizzazione delle azioni.
    # Prova "libero_goal", se non esiste prova "libero_goal_no_noops".
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    # Verifica che la chiave esista nelle statistiche del modello
    assert unnorm_key in model.norm_stats
    cfg.unnorm_key = unnorm_key

    # Carica il processor (tokenizer testo + image preprocessor)
    processor = get_processor(cfg)

    # Verifica l'esistenza del proprio_projector (non lo usa, solo check)
    if cfg.use_proprio:
        try:
            get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
            print("✓ proprio_projector loaded")
        except Exception as e:
            print(f"⚠ proprio_projector: {e}")
            cfg.use_proprio = False

    # Verifica l'esistenza dell'action_head (non lo usa, solo check)
    if cfg.use_l1_regression:
        try:
            get_action_head(cfg, model.llm_dim)
            print("✓ action_head loaded")
        except Exception as e:
            print(f"⚠ action_head: {e}")
            cfg.use_l1_regression = False

    # Calcola la dimensione di resize delle immagini
    resize_size = get_image_resize_size(cfg)
    return model, processor, resize_size


# ─────────────────── visual embedding extraction ───────────────────
def extract_visual_embedding(
    model,
    processor,
    img_pil: Image.Image,
    wrist_pil: Optional[Image.Image],
    prompt: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae SOLO la componente visiva dell'embedding: i projected_patches.

    DIFFERENZA CHIAVE rispetto a extract_embedding() in compare_embeddings:
    - compare_embeddings: fa il forward COMPLETO del LLM e estrae gli hidden states
      dell'ultimo layer (embedding multimodale testo+visione, condizionato).
    - QUESTO: si ferma PRIMA del LLM, dopo vision_backbone + projector.
      Estrae i patch embedding proiettati nello spazio del LLM, ma SENZA
      passare attraverso il transformer. Cattura quindi SOLO l'informazione visiva.

    Il prompt è necessario solo perché il processor lo richiede per costruire
    il dict di input (in particolare pixel_values), ma se use_film=False
    il prompt NON influenza i projected_patches. Questo è ciò che verifichiamo
    nel sanity check.

    Input:
        model     — modello VLA con .vision_backbone e .projector
        processor — callable: processor(prompt, image) → dict tensori
        img_pil   — PIL.Image dell'immagine frontale (agentview)
        wrist_pil — PIL.Image della wrist camera (o None se non usata)
        prompt    — stringa del prompt (necessaria al processor, non influisce sui patch)

    Output (tuple):
        patches_flat : np.ndarray (n_patches, hidden_dim) — tutti i patch embedding proiettati.
                       Ogni riga è un patch (regione dell'immagine) nello spazio del LLM.
                       Es. (576, 4096) per 576 patch con hidden_dim=4096.
        patches_mean : np.ndarray (hidden_dim,) — media aritmetica di tutti i patch.
                       Un singolo vettore che riassume l'intera informazione visiva.
    """
    # Disabilita il calcolo dei gradienti (non servono, risparmia memoria)
    with torch.no_grad():
        # Preprocessa prompt + immagine frontale tramite il processor.
        # Output: dict con 'input_ids', 'attention_mask', 'pixel_values'.
        # .to() sposta su GPU e converte in bfloat16.
        inputs = processor(prompt, img_pil).to(model.device, dtype=torch.bfloat16)

        # Se c'è l'immagine wrist, processala e concatena i pixel_values
        if wrist_pil is not None:
            # Processa la wrist image separatamente
            wrist_inputs = processor(prompt, wrist_pil).to(model.device, dtype=torch.bfloat16)
            # Concatena lungo dim=1 (canali): (1, C, H, W) + (1, C, H, W) → (1, 2C, H, W)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        # ── Vision backbone (ViT: DINOv2 + SigLIP) ──
        # Passa i pixel values attraverso il Vision Transformer.
        # Input: pixel_values (1, C_total, H, W)
        # Output: patch_embeddings — (1, n_patches, vision_dim)
        #   dove n_patches = (H/patch_size)² × num_images, vision_dim = dim interna ViT
        patch_embeddings  = model.vision_backbone(inputs["pixel_values"])

        # ── Projector (MLP) ──
        # Proietta le feature visive dallo spazio del ViT allo spazio del LLM.
        # Input: (1, n_patches, vision_dim)
        # Output: projected_patches — (1, n_patches, hidden_dim) es. (1, 576, 4096)
        # NOTA: questo è il punto in cui ci fermiamo. NON passiamo attraverso il LLM.
        projected_patches = model.projector(patch_embeddings)
        # shape: (1, n_patches, hidden_dim)

        # Rimuove la dimensione batch, stacca dal grafo, sposta su CPU, converte a float32.
        # Risultato: (n_patches, hidden_dim) — matrice con un vettore per patch
        patches_flat = projected_patches.squeeze(0).detach().cpu().float().numpy()  # (n_patches, hidden_dim)
        # Media aritmetica su tutti i patch → un singolo vettore riassuntivo
        # Risultato: (hidden_dim,) — es. (4096,)
        patches_mean = patches_flat.mean(axis=0)                                     # (hidden_dim,)

    return patches_flat, patches_mean


# ─────────────────── first frame helper ───────────────────
def get_first_frame(task, cfg: Cfg, resize_size, task_id: int, task_suite) -> Tuple[Image.Image, Image.Image]:
    """
    Inizializza l'ambiente LIBERO per un task, stabilizza la simulazione,
    cattura il primo frame da entrambe le camere e chiude l'ambiente.

    Input:
        task       — oggetto task LIBERO (da task_suite.get_task(id))
        cfg        — configurazione con env_img_res, model_family, num_steps_wait
        resize_size — dimensione lato target per resize immagini
        task_id    — indice numerico del task (0-9)
        task_suite — oggetto suite benchmark (per ottenere stati iniziali)

    Output (tuple):
        img_pil   — PIL.Image (RGB) dell'immagine frontale agentview, ridimensionata
        wrist_pil — PIL.Image (RGB) dell'immagine wrist camera, ridimensionata
    """
    # Crea l'ambiente di simulazione MuJoCo/robosuite per questo task.
    # Output: (env oggetto ambiente, descrizione_task_str, bddl_path_str)
    env, _, _ = get_libero_env(task, change_command=False, resolution=cfg.env_img_res)
    # Seed fisso a 0 per riproducibilità tra task diversi
    env.seed(0)

    try:
        # Carica gli stati iniziali predefiniti dal benchmark per questo task.
        # Output: lista di np.ndarray, ciascuno è uno stato MuJoCo completo.
        initial_states = task_suite.get_task_init_states(task_id)
        # Reset dell'ambiente alla configurazione base
        env.reset()
        # Imposta lo stato iniziale specifico del benchmark (il primo, indice 0).
        # Output: obs — dict con immagini e dati propriocettivi.
        obs = env.set_init_state(initial_states[0])
    except Exception:
        # Fallback: se gli stati iniziali non sono disponibili, usa il reset di default
        env.reset()
        obs = env.get_observation()

    # Passi di stabilizzazione: esegue num_steps_wait (10) azioni NULLE.
    # Permette alla simulazione fisica di assestarsi (oggetti, robot).
    # env.step() restituisce (obs, reward, done, info)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    # Estrae le immagini dall'osservazione stabilizzata
    # get_libero_image: obs → np.ndarray (H, W, 3) uint8 (vista frontale)
    img       = get_libero_image(obs)
    # get_libero_wrist_image: obs → np.ndarray (H, W, 3) uint8 (vista polso robot)
    wrist_img = get_libero_wrist_image(obs)
    # Chiude l'ambiente per liberare risorse (rendering MuJoCo, memoria)
    env.close()

    # Ridimensiona e converte in PIL.Image RGB (formato atteso dal processor)
    img_pil   = Image.fromarray(resize_image_for_policy(img,       resize_size)).convert("RGB")
    wrist_pil = Image.fromarray(resize_image_for_policy(wrist_img, resize_size)).convert("RGB")
    return img_pil, wrist_pil


# ─────────────────── distance utilities ───────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcola la SIMILARITÀ coseno (NON distanza) tra due vettori.

    Formula: sim_cos(a, b) = (a · b) / (||a|| * ||b||)
    - 1.0 = vettori perfettamente allineati (identici in direzione)
    - 0.0 = vettori ortogonali (nessuna correlazione)
    - -1.0 = vettori opposti

    NOTA: a differenza di cosine_distance in compare_embeddings (che restituisce 1-sim),
    questa funzione restituisce la SIMILARITÀ diretta. Valori alti = più simili.

    Input: a, b: np.ndarray (hidden_dim,). Output: float ∈ [-1, 1].
    """
    # Normalizza a norma unitaria (+ epsilon per evitare divisione per zero)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    # Prodotto scalare di vettori unitari = coseno dell'angolo tra loro
    return float(np.dot(a, b))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcola la distanza euclidea (norma L2 della differenza).

    Formula: d_euc(a, b) = ||a - b||_2 = sqrt(Σ(a_i - b_i)²)

    Input: a, b: np.ndarray (hidden_dim,). Output: float ≥ 0.
    """
    return float(np.linalg.norm(a - b))


# ─────────────────── pretty print matrix ───────────────────
def short(name: str, n: int = 18) -> str:
    """
    Abbrevia il nome di un task per la stampa nella matrice.
    Prende le prime 4 lettere di ogni parola e tronca a n caratteri.

    Input:
        name: str — nome del task con underscore (es. "put_the_wine_bottle_on_top_of_the_cabinet")
        n: int — lunghezza massima dell'abbreviazione (default 18)
    Output:
        str — abbreviazione (es. "put the wine bott")
    """
    # Sostituisce underscore con spazi, divide in parole
    words = name.replace("_", " ").split()
    # Prende le prime 4 lettere di ogni parola e le unisce con spazi
    abbr = " ".join(w[:4] for w in words)
    # Tronca alla lunghezza massima
    return abbr[:n]


def print_matrix(matrix: np.ndarray, labels: List[str], title: str, fmt: str = "{:.4f}"):
    """
    Stampa una matrice NxN formattata come tabella con indici e etichette abbreviate.

    Input:
        matrix — np.ndarray (N, N) con i valori da stampare
        labels — List[str] di N etichette per righe/colonne
        title  — titolo della tabella
        fmt    — formato numerico per i valori (es. "{:.4f}" per 4 decimali)

    Output (su stdout): tabella con header numerici (0-9) e righe etichettate.
    """
    n = len(labels)       # numero di task (10)
    col_w = 10            # larghezza di ogni colonna numerica
    lbl_w = 22            # larghezza della colonna etichetta sulla sinistra
    # Linea separatore di lunghezza proporzionale alla tabella
    print(f"\n{'─'*((n*col_w)+lbl_w+4)}")
    print(f"  {title}")
    print(f"{'─'*((n*col_w)+lbl_w+4)}")
    # Header: spazio vuoto per l'etichetta + indici colonna (0, 1, 2, ..., 9)
    header = f"{'':>{lbl_w}}"
    for i in range(n):
        header += f" {i:>{col_w-1}}"
    print(header)
    # Righe: per ogni task, stampa il suo indice, etichetta abbreviata, poi i valori
    for i in range(n):
        # Costruisce l'etichetta della riga: indice + nome abbreviato
        row = f"  {i:2d} {short(labels[i]):<{lbl_w-4}}"
        # Aggiunge ogni valore della riga formattato secondo fmt
        for j in range(n):
            row += f" {fmt.format(matrix[i, j]):>{col_w-1}}"
        print(row)


# ─────────────────── sanity check ───────────────────
def sanity_check_text_independence(model, processor, img_pil, wrist_pil, prompts: List[str]):
    """
    Verifica che lo stesso frame con prompt diversi produca projected_patches IDENTICI.

    LOGICA:
    Se use_film=False, il blocco visivo (vision_backbone + projector) processa
    solo i pixel_values e NON riceve condizionamento dal testo. Quindi, cambiando
    il prompt ma mantenendo la stessa immagine, i projected_patches devono essere
    ESATTAMENTE uguali (a meno di errori di precisione floating-point ~1e-7).

    Se invece use_film=True, il testo modula le feature visive tramite FiLM
    (Feature-wise Linear Modulation), e quindi i patch sarebbero DIVERSI.

    Questo sanity check è fondamentale perché, se confermato, garantisce che
    le differenze negli embedding multimodali (viste in compare_embeddings)
    derivano ESCLUSIVAMENTE dalla componente linguistica, non da quella visiva.

    Input:
        model     — modello VLA
        processor — processor del modello
        img_pil   — PIL.Image dell'immagine frontale (fissa per tutti i prompt)
        wrist_pil — PIL.Image della wrist camera (o None)
        prompts   — List[str] di prompt diversi da testare sulla stessa immagine

    Output (su stdout):
        Per ogni coppia (prompt_0, prompt_i): max e mean della differenza assoluta
        tra i patch, e verdetto IDENTICI/DIVERSI (soglia 1e-5).
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: stessa immagine, prompt diversi → projected_patches identici?")
    print("=" * 80)

    # Estrae i projected_patches per ogni prompt (stessa immagine)
    all_flat = []
    for p in prompts:
        # Estrae i patch visivi — se use_film=False, il prompt non li influenza
        flat, _ = extract_visual_embedding(model, processor, img_pil, wrist_pil, p)
        all_flat.append(flat)  # (n_patches, hidden_dim)
        print(f"  prompt: {p[:70]}...")  # Stampa i primi 70 char del prompt

    # Confronta ogni prompt con il primo (riferimento)
    ref = all_flat[0]  # Patch del prompt 0 come riferimento
    all_identical = True
    for i, flat in enumerate(all_flat[1:], 1):
        # Differenza elemento per elemento tra i patch di ref e quelli di prompt_i
        max_diff   = float(np.abs(ref - flat).max())   # Massima differenza assoluta
        mean_diff  = float(np.abs(ref - flat).mean())  # Media delle differenze assolute
        # np.allclose: True se tutti gli elementi sono uguali entro tolleranza atol=1e-5
        identical  = np.allclose(ref, flat, atol=1e-5)
        all_identical = all_identical and identical
        status = "✓ IDENTICI" if identical else "✗ DIVERSI"
        print(f"\n  Prompt 0 vs Prompt {i}: {status}")
        print(f"    max|diff|  = {max_diff:.2e}")   # Es. 0.00e+00 se perfettamente uguali
        print(f"    mean|diff| = {mean_diff:.2e}")

    # Conclusione finale
    if all_identical:
        print("\n  ✓ CONFERMATO: il blocco visivo è indipendente dal testo (use_film=False)")
    else:
        print("\n  ✗ ATTENZIONE: il blocco visivo dipende dal testo → FiLM conditioning attivo!")


# ─────────────────── main ───────────────────
def main():
    """
    Funzione principale. Orchestrazione completa dell'analisi:
    1. Carica il modello.
    2. Per ogni task LIBERO Goal (10 task), cattura il primo frame e calcola il visual embedding.
    3. Costruisce matrici 10×10 di cosine similarity e euclidean distance.
    4. Calcola statistiche aggregate sulla matrice (off-diagonal).
    5. Fornisce un'interpretazione automatica dei risultati.
    6. Esegue il sanity check: stesso frame + prompt diversi → patch identici?
    """
    # ── STEP 1: Creazione configurazione e caricamento modello ──
    cfg = Cfg()  # Configurazione con valori di default
    # Carica modello VLA, processor e calcola resize_size
    model, processor, resize_size = load_model(cfg)

    # ── STEP 2: Pre-caricamento della suite benchmark ──
    # Ottiene dizionario {nome_suite: classe_suite}
    benchmark_dict = benchmark.get_benchmark_dict()
    # Istanzia la suite "libero_goal" con i suoi 10 task da cucina
    task_suite     = benchmark_dict[cfg.task_suite_name]()
    # Numero totale di task nella suite (10 per libero_goal)
    n_tasks        = task_suite.n_tasks  # 10

    # Stampa header dell'analisi
    print(f"\n{'='*80}")
    print(f"VISUAL EMBEDDING ANALYSIS — OpenVLA-OFT checkpoint 20000 — LIBERO Goal")
    print(f"{'='*80}")
    # Mostra se FiLM è attivo (influenza l'interpretazione dei risultati)
    print(f"Modello use_film = {cfg.use_film}")
    print(f"Num tasks        = {n_tasks}")
    print(f"{'='*80}\n")

    # ── STEP 3: Estrazione visual embedding per ogni task ──
    # Liste per accumulare i risultati di tutti i task
    mean_embeddings: List[np.ndarray] = []  # Lista di vettori (hidden_dim,) — uno per task
    flat_embeddings: List[np.ndarray] = []  # Lista di matrici (n_patches, hidden_dim) — uno per task
    task_labels: List[str] = []             # Nomi dei task per le etichette della matrice

    # Prompt generico/neutro passato al processor per ottenere i pixel_values.
    # Se use_film=False (come in questa configurazione), il testo del prompt
    # NON influenza i projected_patches, quindi il contenuto è irrilevante.
    # Usiamo un prompt generico uguale per tutti i task per coerenza.
    neutral_prompt = "In: What action should the robot take to perform the task?\nOut:"

    # Itera su tutti i 10 task della suite
    for task_id in range(n_tasks):
        # Ottiene l'oggetto task dal benchmark (contiene BDDL, lingua, ecc.)
        task = task_suite.get_task(task_id)
        # Nome del task dalla lista costante
        task_key = TASK_NAMES[task_id]
        task_labels.append(task_key)

        print(f"[{task_id:2d}] {task_key}")
        # Cattura il primo frame dell'ambiente per questo task
        img_pil, wrist_pil = get_first_frame(task, cfg, resize_size, task_id, task_suite)

        # Passa l'immagine wrist solo se il modello usa 2 immagini
        wrist = wrist_pil if cfg.num_images_in_input > 1 else None
        # Estrae i visual embedding: SOLO blocco visivo (vision_backbone + projector),
        # SENZA forward del LLM.
        # flat: (n_patches, hidden_dim) — tutti i patch
        # mean: (hidden_dim,) — media dei patch
        flat, mean = extract_visual_embedding(model, processor, img_pil, wrist, neutral_prompt)
        flat_embeddings.append(flat)
        mean_embeddings.append(mean)
        # Stampa le dimensioni per debug (es. "patch shape: (576, 4096) | mean shape: (4096,)")
        print(f"     patch shape: {flat.shape}  |  mean shape: {mean.shape}")

    # ── STEP 4: Costruzione matrici di similarità ──
    n = n_tasks  # 10
    # Matrice 10×10 per la cosine similarity tra visual embedding
    cos_matrix = np.zeros((n, n))
    # Matrice 10×10 per la distanza euclidea tra visual embedding
    euc_matrix = np.zeros((n, n))

    # Calcola similarità/distanza per ogni coppia di task (inclusa la diagonale)
    # La diagonale avrà cos_sim=1.0 e euc_dist=0.0 (confronto con se stesso)
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = cosine_sim(mean_embeddings[i], mean_embeddings[j])
            euc_matrix[i, j] = euclidean_dist(mean_embeddings[i], mean_embeddings[j])

    # Stampa la matrice di cosine similarity formattata
    print_matrix(cos_matrix, task_labels,
                 "COSINE SIMILARITY tra visual embedding (mean-pool patches) — 10 task",
                 fmt="{:.4f}")

    # Stampa la matrice di distanza euclidea formattata
    print_matrix(euc_matrix, task_labels,
                 "EUCLIDEAN DISTANCE tra visual embedding (mean-pool patches) — 10 task",
                 fmt="{:.2f}")

    # ── STEP 5: Statistiche off-diagonal ──
    # Crea una maschera booleana che esclude la diagonale (i confronti di un task con se stesso).
    # ~np.eye(n, dtype=bool) → True ovunque tranne la diagonale.
    mask = ~np.eye(n, dtype=bool)
    # Estrae i valori off-diagonal: le N*(N-1) = 90 coppie di task diversi
    off_cos  = cos_matrix[mask]  # Array 1D di 90 valori di cosine similarity
    off_euc  = euc_matrix[mask]  # Array 1D di 90 valori di distanza euclidea

    # Stampa statistiche aggregate: media, deviazione standard, minimo, massimo
    print(f"\n{'─'*80}")
    print("STATISTICHE (coppie off-diagonal, N={})".format(n * (n - 1)))
    print(f"  Cosine Similarity:    mean={off_cos.mean():.4f}  std={off_cos.std():.4f}  "
          f"min={off_cos.min():.4f}  max={off_cos.max():.4f}")
    print(f"  Euclidean Distance:   mean={off_euc.mean():.2f}   std={off_euc.std():.2f}   "
          f"min={off_euc.min():.2f}   max={off_euc.max():.2f}")

    # ── STEP 6: Interpretazione automatica dei risultati ──
    # Basata sulla media della cosine similarity off-diagonal
    mean_cos = off_cos.mean()
    print(f"\n{'─'*80}")
    print("INTERPRETAZIONE:")
    if mean_cos > 0.99:
        # Se la similarity media è >0.99, i visual embedding sono quasi identici.
        # Ciò significa che le scene iniziali dei 10 task producono rappresentazioni
        # visive praticamente indistinguibili. Il modello deve quindi affidarsi
        # ESCLUSIVAMENTE al prompt testuale per sapere quale azione eseguire.
        print("  ● Cosine similarity molto alta (>0.99): i visual embedding sono quasi identici")
        print("    tra tutti i task. Il modello si affida PRINCIPALMENTE al prompt testuale")
        print("    per discriminare quale azione eseguire.")
        print("    → Le differenze di performance su L1/L2/L3 sono attribuibili alla")
        print("      componente LINGUISTICA dell'embedding, non a quella visiva.")
    elif mean_cos > 0.95:
        # Similarity alta ma non estrema: le scene sono simili ma distinguibili.
        print("  ● Cosine similarity alta (>0.95): il blocco visivo produce rappresentazioni")
        print("    simili ma non identiche. Il testo gioca comunque un ruolo dominante.")
    else:
        # Similarity moderata: il blocco visivo differenzia significativamente i task.
        print("  ● Cosine similarity moderata: il blocco visivo differenzia i task.")
        print("    Sia la componente visiva che quella testuale contribuiscono.")

    # ── STEP 7: Sanity check — indipendenza del blocco visivo dal testo ──
    # Usa il task 0 come caso di test: stessa immagine con 4 prompt diversi.
    task0 = task_suite.get_task(0)
    # Cattura il primo frame del task 0
    img0, wrist0 = get_first_frame(task0, cfg, resize_size, 0, task_suite)
    # Esegue il sanity check: se use_film=False, i patch devono essere identici
    # indipendentemente dal prompt usato.
    sanity_check_text_independence(
        model, processor,
        img0,
        wrist0 if cfg.num_images_in_input > 1 else None,
        SANITY_PROMPTS,  # 4 prompt semanticamente diversi
    )

    print(f"\n{'='*80}")
    print("Analisi completata.")


# Entry point: esegue main() solo se lo script è lanciato direttamente (non importato come modulo)
if __name__ == "__main__":
    main()
