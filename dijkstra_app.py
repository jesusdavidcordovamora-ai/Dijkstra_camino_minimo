
"""
Qué se puede hacer:
  * Generar un grafo aleatorio o armarlo a mano (se rechazan los ciclos).
  * Elegir origen y destino con los selectores o con clic sobre un vértice.
  * Arrastrar los vértices para acomodar el grafo.
  * Reproducir el algoritmo, avanzar arista por arista o vértice por vértice,
    volver atrás o saltar a cualquier paso con la barra.
  * Ver la tabla de distancias, la cola de prioridad y el registro.
  * Ver TODOS los caminos mínimos y resaltar cada uno en el grafo.
Atajos: flechas izquierda/derecha, Inicio, Fin y barra espaciadora.
"""

import heapq
import math
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import ttk
from typing import List, NamedTuple, Optional, Tuple

INF = float("inf")


# ============================================================
# LÓGICA DEL GRAFO Y DEL ALGORITMO (sin interfaz)
# ============================================================

class Edge(NamedTuple):
    u: int
    v: int
    w: int


def random_edges(n: int, ties: bool = False) -> List[Edge]:
    """Grafo dirigido sin ciclos: toda arista va de un vértice menor a uno mayor.
    Todo vértice (menos el 0) recibe al menos una conexión de entrada."""
    wmax = 5 if ties else 20
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.35:
                edges[(i, j)] = Edge(i, j, random.randint(1, wmax))
    for j in range(1, n):
        if not any(v == j for (_, v) in edges):
            p = random.randint(0, j - 1)
            edges[(p, j)] = Edge(p, j, random.randint(1, wmax))
    return list(edges.values())


def reaches(edges: List[Edge], start: int, goal: int) -> bool:
    """True si desde 'start' se puede llegar a 'goal' siguiendo las aristas."""
    adj = {}
    for e in edges:
        adj.setdefault(e.u, []).append(e.v)
    seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        if x == goal:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(adj.get(x, []))
    return False


def enumerate_paths(preds, src: int, dst: int, cap: int = 400) -> List[List[int]]:
    """Reconstruye los caminos mínimos yendo de 'dst' hacia atrás hasta 'src'."""
    result, stack = [], [dst]

    def walk(v):
        if len(result) >= cap:
            return
        if v == src:
            result.append(list(reversed(stack)))
            return
        for p in preds[v]:
            stack.append(p)
            walk(p)
            stack.pop()
            if len(result) >= cap:
                return

    walk(dst)
    return result


def count_paths(preds, src: int, dst: int) -> int:
    memo = {}

    def c(v):
        if v == src:
            return 1
        if v not in memo:
            memo[v] = sum(c(p) for p in preds[v])
        return memo[v]

    return c(dst)


@dataclass
class Step:
    kind: str                     
    msg: str                       
    log: str                       
    dist: List[float]
    preds: List[List[int]]
    visited: set
    queue: List[Tuple[int, int]]
    current: Optional[int] = None
    active: Optional[Tuple[int, int]] = None
    outcome: str = ""              # 'better' | 'tie' | 'worse'


@dataclass
class Result:
    steps: List[Step]
    paths: List[List[int]]
    count: int
    distance: float


def fmt(d) -> str:
    return "∞" if d == INF else str(d)


def run_dijkstra(n: int, edges: List[Edge], src: int, dst: int) -> Result:
    """Ejecuta Dijkstra completo y guarda una 'foto' por cada paso, para poder
    avanzar y retroceder en la interfaz."""
    out = [[] for _ in range(n)]
    for e in edges:
        out[e.u].append(e)
    for lst in out:
        lst.sort(key=lambda e: e.v)

    dist = [INF] * n
    dist[src] = 0
    preds = [[] for _ in range(n)]     
    visited = set()
    heap = [(0, src)]
    steps: List[Step] = []

    def snap(kind, msg, log, current=None, active=None, outcome=""):
        queue = sorted((d, v) for d, v in heap if v not in visited and d == dist[v])
        steps.append(Step(kind, msg, log, list(dist), [list(p) for p in preds],
                          set(visited), queue, current, active, outcome))

    snap("init",
         f"Empezamos en el vértice {src}: su distancia es 0 y la de los demás es ∞. "
         f"Buscamos el camino más corto hasta el vértice {dst}.",
         f"Inicio · origen {src}, destino {dst}")

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        outs = out[u]
        extra = (f"Ahora se revisan sus {len(outs)} salida{'s' if len(outs) != 1 else ''}."
                 if outs else "No tiene salidas.")
        snap("visit",
             f"Se visita el vértice {u}, el pendiente más cercano (distancia {d}). {extra}",
             f"Visitando vértice {u} (distancia acumulada = {d})", current=u)

        for e in outs:
            v, nd, old = e.v, dist[u] + e.w, dist[e.v]
            if nd < old:
                dist[v] = nd
                preds[v] = [u]
                heapq.heappush(heap, (nd, v))
                outcome = "better"
            elif nd == old:
                preds[v].append(u)
                outcome = "tie"
            else:
                outcome = "worse"
            head = f"{u} → {v} (peso {e.w}): {dist[u]} + {e.w} = {nd}"
            if outcome == "better":
                msg = f"{head}, mejor que {fmt(old)}. Nueva distancia mínima del vértice {v}."
                log = f"{u} → {v} (peso {e.w}): nueva distancia mínima = {nd}"
            elif outcome == "tie":
                msg = f"{head}, igual que la distancia actual. Se guarda otro camino mínimo hacia el vértice {v}."
                log = f"{u} → {v} (peso {e.w}): otro camino con la misma distancia ({nd})"
            else:
                msg = f"{head}, no mejora la distancia actual ({dist[v]}). Se descarta."
                log = f"{u} → {v} (peso {e.w}): no mejora la distancia actual ({dist[v]})"
            snap("relax", msg, log, current=u, active=(u, v), outcome=outcome)

    reachable = dist[dst] != INF
    paths = enumerate_paths(preds, src, dst) if reachable else []
    count = count_paths(preds, src, dst) if reachable else 0
    if reachable:
        plural = "" if count == 1 else "s"
        msg = (f"Fin del algoritmo. La distancia mínima de {src} a {dst} es {dist[dst]}, "
               f"con {count} camino{plural} mínimo{plural}.")
    else:
        msg = f"Fin del algoritmo. No existe un camino de {src} a {dst}."
    snap("end", msg, "Fin del algoritmo")
    return Result(steps, paths, count, dist[dst])


def compute_layout(n: int, edges: List[Edge]):
    """Dibuja el grafo por capas, de izquierda a derecha.
    Devuelve posiciones normalizadas (0..1), la capa de cada vértice,
    la cantidad de capas y el máximo de vértices en una misma capa."""
    if not edges:
        rows = 4
        cols = math.ceil(n / rows)
        pos = []
        for v in range(n):
            c, r = divmod(v, rows)
            pos.append(((c / (cols - 1)) if cols > 1 else 0.5, (r + 1) / (rows + 1)))
        return pos, [0] * n, cols, rows

    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    deg = [0] * n
    for e in edges:
        adj[e.u].append(e.v)
        radj[e.v].append(e.u)
        deg[e.v] += 1

    layer = [0] * n
    queue = [v for v in range(n) if deg[v] == 0]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            deg[v] -= 1
            if deg[v] == 0:
                queue.append(v)

    L = max(layer) + 1
    cols = [[] for _ in range(L)]
    for v in range(n):
        cols[layer[v]].append(v)

    yf = {}

    def set_y():
        for c in cols:
            for k, v in enumerate(c):
                yf[v] = (k + 1) / (len(c) + 1)

    def bary(v, nb):
        return sum(yf[p] for p in nb[v]) / len(nb[v]) if nb[v] else yf[v]

    set_y()
    for _ in range(4):                      # ordena cada capa para reducir cruces
        for l in range(1, L):
            cols[l].sort(key=lambda v: (bary(v, radj), v))
            set_y()
        for l in range(L - 2, -1, -1):
            cols[l].sort(key=lambda v: (bary(v, adj), v))
            set_y()

    pos = [None] * n
    for l, c in enumerate(cols):
        fx = l / (L - 1) if L > 1 else 0.5
        for v in c:
            pos[v] = (fx, yf[v])
    return pos, layer, L, max(len(c) for c in cols)


# ============================================================
# COLORES Y TEMAS
# ============================================================

LIGHT = dict(bg="#E8EDF0", surface="#F5F8F9", stage="#FBFCFC", ink="#12212A", muted="#566872",
             line="#CAD4DA", grid="#E3EAEE", accent="#0B7A75", edge="#8E9DA6",
             btn_bg="#12212A", btn_fg="#F5F8F9", cur="#E08700", org="#2F7BD8", dst="#E0553F",
             vis="#2E9E5B", idle="#8A97A0", path="#D1242F", on_cur="#171004", on_path="#FFFFFF")
DARK = dict(bg="#0B1419", surface="#111D24", stage="#0D171D", ink="#E5EDF1", muted="#8DA0AB",
            line="#243640", grid="#15232B", accent="#3CC9BF", edge="#5C6F7A",
            btn_bg="#E5EDF1", btn_fg="#0B1419", cur="#FFB020", org="#5BA3F5", dst="#FF7A63",
            vis="#4CC27F", idle="#6B7A84", path="#FF5C66", on_cur="#171004", on_path="#1A0B0D")


def mix(c1: str, c2: str, t: float) -> str:
    """Mezcla dos colores #rrggbb: t de c1 y (1 - t) de c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x * t + y * (1 - t)) for x, y in zip(a, b))


def pick_font(root, candidates, fallback):
    available = set(tkfont.families(root))
    for c in candidates:
        if c in available:
            return c
    return fallback


def rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


# ============================================================
# APLICACIÓN
# ============================================================

class App:
    PADX, PADY = 46, 42

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Dijkstra paso a paso")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1380, sw - 60)}x{min(940, sh - 90)}+30+20")
        root.minsize(1000, 640)

        self.body = pick_font(root, ("Segoe UI", "Inter", "Helvetica Neue", "Cantarell", "Ubuntu", "DejaVu Sans"), "Helvetica")
        self.mono = pick_font(root, ("JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New"), "Courier")
        self.f_label = tkfont.Font(family=self.mono, size=9, weight="bold")

        # estado
        self.n = 9
        self.mode = tk.StringVar(value="random")
        self.ties = tk.BooleanVar(value=False)
        self.dark = tk.BooleanVar(value=False)
        self.edges: List[Edge] = []
        self.src, self.dst = 0, 8
        self.pick = "src"
        self.run: Optional[Result] = None
        self.steps: List[Step] = []
        self.i = 0
        self.last_painted = -1
        self.last_final = None
        self.playing = False
        self.play_job = None
        self.hover = None
        self.pinned = None
        self.pos, self.layer, self.L, self.maxcount = [], [], 1, 1
        self.r = 18
        self.drag = None
        self.halo_on = True
        self.flash_job = None
        self.row_tag = {}
        self.msg_err = False
        self._lock = False
        self._cfg_job = None

        self.pal = LIGHT
        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self.build_ui()
        self.apply_theme()
        self.bind_keys()

        self.edges = random_edges(self.n, False)
        self.refresh_selects()
        self.root.update_idletasks()
        self.apply_graph()
        self.pulse()

    # ---------- construcción de la interfaz ----------
    def build_ui(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(2, weight=1)

        # Encabezado
        head = ttk.Frame(r, padding=(20, 14, 20, 0))
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text="CAMINO MÍNIMO · GRAFO DIRIGIDO SIN CICLOS", style="Kicker.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(head, text="Dijkstra paso a paso", style="Title.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(head, text="Modo oscuro", variable=self.dark, command=self.on_theme).grid(row=0, column=1, rowspan=2, sticky="e")

        # Configuración
        setup = ttk.Frame(r, style="Card.TFrame", padding=(14, 12))
        setup.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 12))
        for c in range(5):
            setup.columnconfigure(c, weight=1, uniform="setup")

        f = ttk.Frame(setup, style="Inner.TFrame")
        f.grid(row=0, column=0, sticky="ew", padx=(0, 16))
        self.lbl_nodes = ttk.Label(f, text="VÉRTICES  9", style="Cap.TLabel")
        self.lbl_nodes.pack(anchor="w")
        self.scale_nodes = ttk.Scale(f, from_=7, to=16, orient="horizontal", style="Card.Horizontal.TScale", command=self.on_nodes)
        self.scale_nodes.set(self.n)
        self.scale_nodes.pack(fill="x", pady=(8, 0))

        f = ttk.Frame(setup, style="Inner.TFrame")
        f.grid(row=0, column=1, sticky="ew", padx=(0, 16))
        ttk.Label(f, text="GRAFO", style="Cap.TLabel").pack(anchor="w")
        row = ttk.Frame(f, style="Inner.TFrame")
        row.pack(anchor="w", pady=(4, 0))
        ttk.Radiobutton(row, text="Aleatorio", value="random", variable=self.mode, style="Card.TRadiobutton", command=self.on_mode).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(row, text="Manual", value="manual", variable=self.mode, style="Card.TRadiobutton", command=self.on_mode).pack(side="left")

        f = ttk.Frame(setup, style="Inner.TFrame")
        f.grid(row=0, column=2, sticky="ew", padx=(0, 16))
        ttk.Label(f, text="ORIGEN", style="Cap.TLabel").pack(anchor="w")
        self.cb_src = ttk.Combobox(f, state="readonly", width=6)
        self.cb_src.pack(anchor="w", pady=(4, 0))
        self.cb_src.bind("<<ComboboxSelected>>", lambda e: self.set_endpoint("src", int(self.cb_src.get())))

        f = ttk.Frame(setup, style="Inner.TFrame")
        f.grid(row=0, column=3, sticky="ew", padx=(0, 16))
        ttk.Label(f, text="DESTINO", style="Cap.TLabel").pack(anchor="w")
        self.cb_dst = ttk.Combobox(f, state="readonly", width=6)
        self.cb_dst.pack(anchor="w", pady=(4, 0))
        self.cb_dst.bind("<<ComboboxSelected>>", lambda e: self.set_endpoint("dst", int(self.cb_dst.get())))

        f = ttk.Frame(setup, style="Inner.TFrame")
        f.grid(row=0, column=4, sticky="ew")
        ttk.Button(f, text="Grafo aleatorio nuevo", command=self.on_regen).pack(anchor="w")
        ttk.Checkbutton(f, text="Más empates (pesos 1–5)", variable=self.ties, style="Card.TCheckbutton", command=self.on_ties).pack(anchor="w", pady=(6, 0))

        # Panel manual
        self.manual = ttk.Frame(setup, style="Inner.TFrame")
        self.manual.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        m = self.manual
        ttk.Label(m, text="DESDE", style="Cap.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(m, text="HACIA", style="Cap.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Label(m, text="PESO", style="Cap.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.cb_from = ttk.Combobox(m, state="readonly", width=5)
        self.cb_to = ttk.Combobox(m, state="readonly", width=5)
        self.sp_w = ttk.Spinbox(m, from_=0, to=99, width=5)
        self.sp_w.set(5)
        self.cb_from.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.cb_to.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(4, 0))
        self.sp_w.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(4, 0))
        ttk.Button(m, text="Agregar conexión", style="Primary.TButton", command=self.add_edge).grid(row=1, column=3, padx=(14, 0), pady=(4, 0))
        ttk.Button(m, text="Vaciar grafo", command=self.clear_edges).grid(row=1, column=4, padx=(8, 0), pady=(4, 0))
        self.msg = ttk.Label(m, text="Las conexiones que formen un ciclo se rechazan. Si repites una, se actualiza su peso.", style="Muted.TLabel")
        self.msg.grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 4))
        lf = ttk.Frame(m, style="Inner.TFrame")
        lf.grid(row=3, column=0, columnspan=6, sticky="ew")
        self.lb_edges = tk.Listbox(lf, height=3, width=26, exportselection=False, relief="flat", highlightthickness=1, borderwidth=0, font=(self.mono, 10))
        self.lb_edges.pack(side="left")
        self.lb_edges.bind("<Double-Button-1>", lambda e: self.remove_selected_edge())
        ttk.Button(lf, text="Quitar seleccionada", command=self.remove_selected_edge).pack(side="left", padx=(10, 0), anchor="n")
        ttk.Label(lf, text="También puedes quitar una conexión con doble clic.", style="Muted.TLabel").pack(side="left", padx=(12, 0), anchor="n")
        self.manual.grid_remove()

        # Zona principal
        main = ttk.Frame(r, padding=(20, 0, 20, 16))
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=410)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        # -- escenario
        stage = ttk.Frame(left, style="Card.TFrame")
        stage.grid(row=0, column=0, sticky="nsew")
        stage.columnconfigure(0, weight=1)
        stage.rowconfigure(1, weight=1)
        top = ttk.Frame(stage, style="Inner.TFrame", padding=(14, 10, 14, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="GRAFO", style="Cap.TLabel").grid(row=0, column=0, sticky="w")
        self.hint = ttk.Label(top, text="", style="Muted.TLabel")
        self.hint.grid(row=0, column=1, sticky="e")

        self.canvas = tk.Canvas(stage, highlightthickness=0, borderwidth=0, width=640, height=380)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover_node)

        leg = ttk.Frame(stage, style="Inner.TFrame", padding=(14, 8, 14, 0))
        leg.grid(row=2, column=0, sticky="ew")
        self.legend = []
        items = [("●", "org", "Origen"), ("●", "dst", "Destino"), ("●", "cur", "Vértice actual"),
                 ("●", "vis", "Visitado"), ("●", "idle", "Pendiente"),
                 ("━", "cur", "Arista en revisión"), ("━", "accent", "Mejor ruta conocida"), ("━", "path", "Camino mínimo")]
        for k, (glyph, key, text) in enumerate(items):
            g = ttk.Label(leg, text=glyph, style="Legend.TLabel")
            g.grid(row=k // 4, column=(k % 4) * 2, sticky="w")
            self.legend.append((g, key))
            ttk.Label(leg, text=text, style="Muted.TLabel").grid(row=k // 4, column=(k % 4) * 2 + 1, sticky="w", padx=(3, 14))
        ttk.Label(stage, text="El número sobre cada vértice es la distancia mínima conocida hasta él (∞ mientras no se alcance).",
                  style="Muted.TLabel", padding=(14, 4, 14, 8)).grid(row=3, column=0, sticky="w")

        # -- reproductor
        pl = ttk.Frame(stage, style="Inner.TFrame", padding=(14, 8, 14, 12))
        pl.grid(row=4, column=0, sticky="ew")
        pl.columnconfigure(0, weight=1)
        self.scrub = ttk.Scale(pl, from_=0, to=1, orient="horizontal", style="Card.Horizontal.TScale", command=self.on_scrub)
        self.scrub.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.step_lbl = ttk.Label(pl, text="Paso 1 de 1", style="Mono.TLabel")
        self.step_lbl.grid(row=0, column=1, sticky="e")
        bar = ttk.Frame(pl, style="Inner.TFrame")
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.b_reset = ttk.Button(bar, text="Reiniciar", command=lambda: (self.stop(), self.go(0)))
        self.b_prev = ttk.Button(bar, text="‹ Anterior", command=lambda: (self.stop(), self.go(self.i - 1)))
        self.b_play = ttk.Button(bar, text="▶ Reproducir", style="Primary.TButton", command=self.toggle_play)
        self.b_next = ttk.Button(bar, text="Siguiente ›", command=lambda: (self.stop(), self.go(self.i + 1)))
        self.b_vertex = ttk.Button(bar, text="Siguiente vértice", command=lambda: (self.stop(), self.next_vertex()))
        self.b_end = ttk.Button(bar, text="Ir al resultado", command=lambda: (self.stop(), self.go(len(self.steps) - 1)))
        for b in (self.b_reset, self.b_prev, self.b_play, self.b_next, self.b_vertex, self.b_end):
            b.pack(side="left", padx=(0, 6))
        self.speed = ttk.Combobox(bar, state="readonly", width=7, values=("Lenta", "Normal", "Rápida"))
        self.speed.set("Normal")
        self.speed.pack(side="right")
        ttk.Label(bar, text="Velocidad", style="Muted.TLabel").pack(side="right", padx=(0, 6))

        # -- resultado
        res = ttk.Frame(left, style="Card.TFrame", padding=(14, 10, 14, 12))
        res.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        res.columnconfigure(0, weight=1)
        ttk.Label(res, text="CAMINOS MÍNIMOS", style="Cap.TLabel").grid(row=0, column=0, sticky="w")

        self.res_wait = ttk.Frame(res, style="Inner.TFrame")
        self.res_wait.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(self.res_wait, text="Aparecen cuando el algoritmo termina.", style="Muted.TLabel").pack(side="left")
        ttk.Button(self.res_wait, text="Ver el resultado ahora", command=lambda: (self.stop(), self.go(len(self.steps) - 1))).pack(side="left", padx=12)

        self.res_done = ttk.Frame(res, style="Inner.TFrame")
        self.res_done.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.res_done.columnconfigure(2, weight=1)
        self.lbl_dist_cap = ttk.Label(self.res_done, text="Distancia mínima", style="Muted.TLabel")
        self.lbl_dist_cap.grid(row=0, column=0, sticky="w", padx=(0, 30))
        self.lbl_dist = ttk.Label(self.res_done, text="0", style="Big.TLabel")
        self.lbl_dist.grid(row=1, column=0, sticky="w", padx=(0, 30))
        ttk.Label(self.res_done, text="Caminos mínimos", style="Muted.TLabel").grid(row=0, column=1, sticky="w")
        self.lbl_count = ttk.Label(self.res_done, text="0", style="Big.TLabel")
        self.lbl_count.grid(row=1, column=1, sticky="w", padx=(0, 30))
        self.lbl_note = ttk.Label(self.res_done, text="", style="Muted.TLabel", wraplength=420, justify="left")
        self.lbl_note.grid(row=0, column=2, rowspan=2, sticky="w")
        lf = ttk.Frame(self.res_done, style="Inner.TFrame")
        lf.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        lf.columnconfigure(0, weight=1)
        self.lb_paths = tk.Listbox(lf, height=4, width=10, exportselection=False, relief="flat", highlightthickness=1, borderwidth=0, activestyle="none", font=(self.mono, 10))
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.lb_paths.yview)
        self.lb_paths.configure(yscrollcommand=sb.set)
        self.lb_paths.grid(row=0, column=0, sticky="ew")
        sb.grid(row=0, column=1, sticky="ns")
        self.lb_paths.bind("<Motion>", self.on_path_hover)
        self.lb_paths.bind("<Leave>", self.on_path_leave)
        self.lb_paths.bind("<<ListboxSelect>>", self.on_path_select)
        self.btn_all = ttk.Button(self.res_done, text="Ver todos juntos", command=self.show_all_paths)
        self.btn_all.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.res_done.grid_remove()

        # -- panel lateral
        side = ttk.Frame(main, style="Card.TFrame")
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(7, weight=1)

        ttk.Label(side, text="PASO ACTUAL", style="Cap.TLabel", padding=(14, 12, 14, 6)).grid(row=0, column=0, sticky="w")
        box = ttk.Frame(side, style="Inner.TFrame", height=92)
        box.grid(row=1, column=0, sticky="ew", padx=14)
        box.grid_propagate(False)
        box.columnconfigure(0, weight=1)
        self.status = ttk.Label(box, text="", style="Status.TLabel", wraplength=370, justify="left")
        self.status.grid(row=0, column=0, sticky="nw")

        ttk.Separator(side).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(side, text="DISTANCIAS", style="Cap.TLabel", padding=(14, 10, 14, 6)).grid(row=3, column=0, sticky="w")
        self.tree = ttk.Treeview(side, columns=("v", "d", "p", "s"), show="headings", selectmode="none", height=9)
        for col, text, w, anchor in (("v", "VÉRTICE", 100, "w"), ("d", "DIST.", 56, "e"), ("p", "VIENE DE", 90, "w"), ("s", "ESTADO", 90, "w")):
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=w, anchor=anchor, stretch=(col == "s"))
        self.tree.grid(row=4, column=0, sticky="ew", padx=14)

        ttk.Label(side, text="COLA DE PRIORIDAD  (distancia, vértice)", style="Cap.TLabel", padding=(14, 12, 14, 4)).grid(row=5, column=0, sticky="w")
        self.queue = ttk.Label(side, text="", style="Queue.TLabel", wraplength=370, justify="left")
        self.queue.grid(row=6, column=0, sticky="w", padx=14)

        lg = ttk.Frame(side, style="Inner.TFrame", padding=(14, 12, 14, 12))
        lg.grid(row=7, column=0, sticky="nsew")
        lg.columnconfigure(0, weight=1)
        lg.rowconfigure(1, weight=1)
        ttk.Label(lg, text="REGISTRO", style="Cap.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.log = tk.Text(lg, height=6, width=10, wrap="word", relief="flat", highlightthickness=1, borderwidth=0, state="disabled", font=(self.mono, 9), padx=6, pady=4)
        lsb = ttk.Scrollbar(lg, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        self.log.grid(row=1, column=0, sticky="nsew")
        lsb.grid(row=1, column=1, sticky="ns")

    # ---------- tema ----------
    def apply_theme(self):
        p = self.pal = DARK if self.dark.get() else LIGHT
        s = self.style
        body, mono = self.body, self.mono
        self.root.configure(bg=p["bg"])
        s.configure(".", background=p["bg"], foreground=p["ink"], font=(body, 10), bordercolor=p["line"],
                    lightcolor=p["bg"], darkcolor=p["bg"], troughcolor=p["line"], focuscolor=p["accent"])
        s.configure("TFrame", background=p["bg"])
        s.configure("Card.TFrame", background=p["surface"], bordercolor=p["line"], borderwidth=1, relief="solid")
        s.configure("Inner.TFrame", background=p["surface"], borderwidth=0, relief="flat")
        s.configure("TLabel", background=p["bg"], foreground=p["ink"])
        s.configure("Kicker.TLabel", background=p["bg"], foreground=p["muted"], font=(mono, 9))
        s.configure("Title.TLabel", background=p["bg"], foreground=p["ink"], font=(body, 28, "bold"))
        s.configure("Cap.TLabel", background=p["surface"], foreground=p["muted"], font=(mono, 8, "bold"))
        s.configure("Muted.TLabel", background=p["surface"], foreground=p["muted"], font=(body, 9))
        s.configure("Mono.TLabel", background=p["surface"], foreground=p["muted"], font=(mono, 9))
        s.configure("Legend.TLabel", background=p["surface"], font=(body, 11))
        s.configure("Status.TLabel", background=p["surface"], foreground=p["ink"], font=(body, 11))
        s.configure("Queue.TLabel", background=p["surface"], foreground=p["ink"], font=(mono, 10))
        s.configure("Big.TLabel", background=p["surface"], foreground=p["ink"], font=(mono, 28, "bold"))
        for name in ("Card.TCheckbutton", "Card.TRadiobutton"):
            s.configure(name, background=p["surface"], foreground=p["ink"], indicatorbackground=p["stage"],
                        indicatorforeground=p["accent"], focuscolor=p["surface"])
            s.map(name, background=[("active", p["surface"])], indicatorbackground=[("selected", p["stage"])])
        s.configure("TCheckbutton", background=p["bg"], foreground=p["ink"], indicatorbackground=p["stage"], indicatorforeground=p["accent"])
        s.map("TCheckbutton", background=[("active", p["bg"])])
        s.configure("TButton", background=p["stage"], foreground=p["ink"], bordercolor=p["line"], padding=(10, 6), relief="flat", font=(body, 10, "bold"))
        s.map("TButton", background=[("active", p["grid"]), ("disabled", p["stage"])], foreground=[("disabled", p["muted"])],
              bordercolor=[("active", p["ink"])])
        s.configure("Primary.TButton", background=p["btn_bg"], foreground=p["btn_fg"], bordercolor=p["btn_bg"])
        s.map("Primary.TButton", background=[("active", mix(p["btn_bg"], p["bg"], .85)), ("disabled", p["btn_bg"])],
              foreground=[("disabled", p["btn_fg"])], bordercolor=[("active", p["btn_bg"])])
        s.configure("TCombobox", fieldbackground=p["stage"], background=p["stage"], foreground=p["ink"], arrowcolor=p["ink"],
                    bordercolor=p["line"], selectbackground=p["stage"], selectforeground=p["ink"], padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", p["stage"])], foreground=[("readonly", p["ink"])],
              selectbackground=[("readonly", p["stage"])], selectforeground=[("readonly", p["ink"])])
        s.configure("TSpinbox", fieldbackground=p["stage"], background=p["stage"], foreground=p["ink"], arrowcolor=p["ink"], bordercolor=p["line"], padding=4)
        s.configure("Card.Horizontal.TScale", background=p["surface"], troughcolor=p["line"], bordercolor=p["line"])
        s.configure("Treeview", background=p["stage"], fieldbackground=p["stage"], foreground=p["ink"], rowheight=24,
                    bordercolor=p["line"], font=(mono, 9))
        s.configure("Treeview.Heading", background=p["surface"], foreground=p["muted"], font=(mono, 8, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", p["stage"])], foreground=[("selected", p["ink"])])
        s.map("Treeview.Heading", background=[("active", p["surface"])])
        s.configure("Vertical.TScrollbar", background=p["line"], troughcolor=p["surface"], arrowcolor=p["muted"], bordercolor=p["surface"])
        s.configure("TSeparator", background=p["line"])

        # tags de la tabla
        t = self.tree
        t.tag_configure("cur", background=mix(p["cur"], p["stage"], .30), foreground=p["ink"])
        t.tag_configure("vis", foreground=mix(p["vis"], p["ink"], .75))
        t.tag_configure("que", foreground=p["accent"])
        t.tag_configure("idle", foreground=p["muted"])
        t.tag_configure("flash", background=mix(p["accent"], p["stage"], .38))

        # widgets clásicos de tk
        self.canvas.configure(bg=p["stage"])
        for w in (self.lb_edges, self.lb_paths):
            w.configure(bg=p["stage"], fg=p["ink"], highlightbackground=p["line"], highlightcolor=p["line"],
                        selectbackground=mix(p["path"], p["stage"], .25), selectforeground=p["ink"])
        self.log.configure(bg=p["stage"], fg=p["ink"], highlightbackground=p["line"], highlightcolor=p["line"])
        lg = self.log
        lg.tag_configure("k-visit", foreground=mix(p["cur"], p["ink"], .8), font=(mono, 9, "bold"), spacing1=4)
        lg.tag_configure("k-better", foreground=p["accent"])
        lg.tag_configure("k-tie", foreground=p["org"])
        lg.tag_configure("k-worse", foreground=p["muted"])
        lg.tag_configure("k-init", foreground=p["ink"], font=(mono, 9, "bold"))
        lg.tag_configure("k-end", foreground=p["ink"], font=(mono, 9, "bold"))
        lg.tag_configure("now", background=mix(p["accent"], p["stage"], .22))
        for lbl, key in self.legend:
            lbl.configure(foreground=p[key])
        self.msg.configure(foreground=p["path"] if self.msg_err else p["muted"])
        for cb in (self.cb_src, self.cb_dst, self.cb_from, self.cb_to, self.speed):
            try:
                pop = str(cb.tk.call("ttk::combobox::PopdownWindow", cb))
                cb.tk.call(pop + ".f.l", "configure", "-background", p["stage"], "-foreground", p["ink"],
                           "-selectbackground", p["accent"], "-selectforeground", p["btn_fg"])
            except tk.TclError:
                pass

    def on_theme(self):
        self.apply_theme()
        self.draw_background()
        if self.steps:
            self.paint()

    # ---------- selects, mensajes, lista de conexiones ----------
    def refresh_selects(self):
        vals = [str(v) for v in range(self.n)]
        for cb in (self.cb_src, self.cb_dst, self.cb_from, self.cb_to):
            cb.configure(values=vals)
        self.cb_src.set(str(self.src))
        self.cb_dst.set(str(self.dst))
        if not self.cb_from.get() or int(self.cb_from.get()) >= self.n:
            self.cb_from.set("0")
        if not self.cb_to.get() or int(self.cb_to.get()) >= self.n:
            self.cb_to.set(str(min(1, self.n - 1)))

    def set_msg(self, text, err=False):
        self.msg_err = err
        self.msg.configure(text=text, foreground=self.pal["path"] if err else self.pal["muted"])

    def refresh_edge_list(self):
        self.lb_edges.delete(0, "end")
        self._edge_order = sorted(range(len(self.edges)), key=lambda k: (self.edges[k].u, self.edges[k].v))
        if not self.edges:
            self.lb_edges.insert("end", "(sin conexiones todavía)")
            return
        for k in self._edge_order:
            e = self.edges[k]
            self.lb_edges.insert("end", f"{e.u} → {e.v}   peso {e.w}")

    def update_hint(self):
        nxt = "origen" if self.pick == "src" else "destino"
        self.hint.configure(text=f"Clic en un vértice para fijar el {nxt}. Arrástralo para acomodarlo.")

    # ---------- acciones de configuración ----------
    def on_nodes(self, val):
        if self._lock:
            return
        v = int(round(float(val)))
        if v == self.n:
            return
        self.n = v
        self.lbl_nodes.configure(text=f"VÉRTICES  {v}")
        if self.mode.get() == "random":
            self.edges = random_edges(self.n, self.ties.get())
            self.src, self.dst = 0, self.n - 1
        else:
            self.edges = [e for e in self.edges if e.u < self.n and e.v < self.n]
            self.src = self.src if self.src < self.n else 0
            self.dst = self.dst if self.dst < self.n else self.n - 1
        self.refresh_selects()
        self.apply_graph()

    def on_mode(self):
        if self.mode.get() == "random":
            self.manual.grid_remove()
            self.edges = random_edges(self.n, self.ties.get())
            self.apply_graph()
        else:
            self.manual.grid()
            self.refresh_edge_list()

    def on_regen(self):
        self.edges = random_edges(self.n, self.ties.get())
        self.apply_graph()

    def on_ties(self):
        if self.mode.get() == "random":
            self.edges = random_edges(self.n, self.ties.get())
            self.apply_graph()

    def set_endpoint(self, which, v):
        if which == "src":
            self.src = v
        else:
            self.dst = v
        self.start_run()

    def add_edge(self):
        try:
            u, v, w = int(self.cb_from.get()), int(self.cb_to.get()), int(float(self.sp_w.get()))
        except ValueError:
            self.set_msg("Revisa los datos: el peso debe ser un número entero.", True)
            return
        if w < 0:
            self.set_msg("El peso no puede ser negativo.", True)
            return
        if u == v:
            self.set_msg("Un vértice no puede conectarse consigo mismo.", True)
            return
        if reaches(self.edges, v, u):
            self.set_msg(f"La conexión {u} → {v} formaría un ciclo, así que no se agregó.", True)
            return
        for k, e in enumerate(self.edges):
            if e.u == u and e.v == v:
                self.edges[k] = Edge(u, v, w)
                self.set_msg(f"Se actualizó el peso de {u} → {v} a {w}.")
                break
        else:
            self.edges.append(Edge(u, v, w))
            self.set_msg(f"Se agregó {u} → {v} con peso {w}.")
        self.apply_graph()

    def clear_edges(self):
        self.edges = []
        self.set_msg("Grafo vacío. Agrega conexiones para empezar.")
        self.apply_graph()

    def remove_selected_edge(self):
        sel = self.lb_edges.curselection()
        if not sel or not self.edges:
            return
        e = self.edges.pop(self._edge_order[sel[0]])
        self.set_msg(f"Se quitó la conexión {e.u} → {e.v}.")
        self.apply_graph()

    # ---------- ejecución del algoritmo ----------
    def apply_graph(self):
        self.pos, self.layer, self.L, self.maxcount = compute_layout(self.n, self.edges)
        self.refresh_edge_list()
        self.update_hint()
        self.start_run()

    def start_run(self):
        self.stop()
        self.run = run_dijkstra(self.n, self.edges, self.src, self.dst)
        self.steps = self.run.steps
        self.i = 0
        self.last_painted = -1
        self.hover = self.pinned = None
        self.last_final = None
        self.ensure_rows()
        self.paint()

    def ensure_rows(self):
        kids = self.tree.get_children()
        if len(kids) != self.n:
            if kids:
                self.tree.delete(*kids)
            for v in range(self.n):
                self.tree.insert("", "end", iid=str(v), values=("", "", "", ""))
            self.tree.configure(height=self.n)

    # ---------- geometría ----------
    def size(self):
        return max(self.canvas.winfo_width(), 300), max(self.canvas.winfo_height(), 240)

    def node_radius(self):
        W, H = self.size()
        if not self.edges:
            return 19
        vs = (H - 2 * self.PADY) / (self.maxcount + 1)
        dx = (W - 2 * self.PADX) / (self.L - 1) if self.L > 1 else 1e9
        return max(11, min(19, vs * 0.4, dx * 0.42))

    def px(self, v):
        W, H = self.size()
        fx, fy = self.pos[v]
        return self.PADX + fx * (W - 2 * self.PADX), self.PADY + fy * (H - 2 * self.PADY)

    def edge_geom(self, e):
        r = self.r
        ax, ay = self.px(e.u)
        bx, by = self.px(e.v)
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) or 1.0
        span = abs(self.layer[e.v] - self.layer[e.u])
        bend = 0.0
        if span > 1:
            bend = (1 if (e.u + e.v) % 2 else -1) * min(70, 10 * span + ln * 0.06)
        nx_, ny_ = -dy / ln, dx / ln
        cx = (ax + bx) / 2 + nx_ * bend * 2
        cy = (ay + by) / 2 + ny_ * bend * 2
        ux, uy = cx - ax, cy - ay
        ul = math.hypot(ux, uy) or 1.0
        sx, sy = ax + ux / ul * r, ay + uy / ul * r
        vx, vy = bx - cx, by - cy
        vl = math.hypot(vx, vy) or 1.0
        ex, ey = bx - vx / vl * (r + 2), by - vy / vl * (r + 2)
        t = 0.35 + ((e.u * 7 + e.v * 13) % 5) * 0.075
        lx = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t * t * ex
        ly = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t * t * ey
        return sx, sy, cx, cy, ex, ey, lx, ly

    def node_at(self, x, y):
        for v in range(self.n):
            vx, vy = self.px(v)
            if math.hypot(x - vx, y - vy) <= self.r + 3:
                return v
        return None

    # ---------- dibujo ----------
    def draw_background(self):
        c = self.canvas
        c.delete("bg")
        W, H = self.size()
        col = self.pal["grid"]
        for x in range(0, W, 28):
            c.create_line(x, 0, x, H, fill=col, tags="bg")
        for y in range(0, H, 28):
            c.create_line(0, y, W, y, fill=col, tags="bg")
        c.tag_lower("bg")

    def on_canvas_configure(self, _evt):
        if self._cfg_job:
            self.root.after_cancel(self._cfg_job)
        self._cfg_job = self.root.after(60, self._after_resize)

    def _after_resize(self):
        self._cfg_job = None
        self.draw_background()
        if self.steps:
            self.draw_graph()

    def draw_graph(self):
        c, p = self.canvas, self.pal
        c.delete("fg")
        if not self.steps or not self.pos:
            return
        self.r = r = self.node_radius()
        s = self.steps[self.i]
        run = self.run
        final = s.kind == "end"
        show_paths = final and bool(run.paths)
        sel = self.hover if self.hover is not None else self.pinned
        has_sel = show_paths and sel is not None and 0 <= sel < len(run.paths)
        keys = set()
        if show_paths:
            for pth in ([run.paths[sel]] if has_sel else run.paths):
                keys.update(zip(pth, pth[1:]))

        order = {"dim": 0, "idle": 1, "tree": 2, "active": 3, "path": 3}
        items = []
        for e in self.edges:
            if show_paths:
                st = "path" if (e.u, e.v) in keys else ("dim" if has_sel else "idle")
            elif s.active == (e.u, e.v):
                st = "active"
            elif e.u in s.preds[e.v]:
                st = "tree"
            else:
                st = "idle"
            items.append((order[st], e, st))
        items.sort(key=lambda t: t[0])

        style = {"idle": (p["edge"], 1.6), "tree": (p["accent"], 2.4), "active": (p["cur"], 3.8),
                 "path": (p["path"], 4.2), "dim": (mix(p["edge"], p["stage"], .3), 1.4)}
        geoms = []
        for _, e, st in items:
            g = self.edge_geom(e)
            geoms.append((e, st, g))
            color, width = style[st]
            c.create_line(g[0], g[1], g[2], g[3], g[4], g[5], smooth=True, fill=color, width=width,
                          arrow=tk.LAST, arrowshape=(11, 13, 5), capstyle="round", tags="fg")

        for e, st, g in geoms:
            txt = str(e.w)
            wd = self.f_label.measure(txt) + 12
            fill, outline, fg = p["stage"], p["line"], p["ink"]
            if st == "tree":
                outline = p["accent"]
            elif st == "active":
                fill, outline, fg = p["cur"], p["cur"], p["on_cur"]
            elif st == "path":
                fill, outline, fg = p["path"], p["path"], p["on_path"]
            elif st == "dim":
                fg = p["muted"]
            lx, ly = g[6], g[7]
            rounded_rect(c, lx - wd / 2, ly - 9, lx + wd / 2, ly + 9, 8, fill=fill, outline=outline, tags="fg")
            c.create_text(lx, ly, text=txt, fill=fg, font=self.f_label, tags="fg")

        fnode = (self.mono, max(8, int(r * 0.62)), "bold")
        fsmall = (self.mono, 9, "bold")
        ftag = (self.mono, 7, "bold")
        for v in range(self.n):
            x, y = self.px(v)
            if v == s.current:
                st, col = "cur", p["cur"]
            elif v == self.src:
                st, col = "org", p["org"]
            elif v == self.dst:
                st, col = "dst", p["dst"]
            elif v in s.visited:
                st, col = "vis", p["vis"]
            else:
                st, col = "idle", p["idle"]
            if st == "cur":
                halo = c.create_oval(x - r - 7, y - r - 7, x + r + 7, y + r + 7, outline=col, width=2, tags=("fg", "halo"))
                c.itemconfigure(halo, state="normal" if self.halo_on else "hidden")
            fill = mix(col, p["stage"], .55 if st == "cur" else .18)
            c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=col, width=3.4 if st == "cur" else 2.4, tags="fg")
            c.create_text(x, y, text=str(v), fill=p["ink"], font=fnode, tags="fg")
            d = s.dist[v]
            self.halo_text(x, y - r - 9, "∞" if d == INF else str(d), fsmall)
            tag = ("ORIGEN · DESTINO" if v == self.src and v == self.dst else
                   "ORIGEN" if v == self.src else "DESTINO" if v == self.dst else "")
            if tag:
                self.halo_text(x, y + r + 10, tag, ftag)

    def halo_text(self, x, y, text, font):
        """Texto con un pequeño contorno del color del fondo, para que se lea sobre las aristas."""
        c, p = self.canvas, self.pal
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            c.create_text(x + dx, y + dy, text=text, fill=p["stage"], font=font, tags="fg")
        c.create_text(x, y, text=text, fill=p["muted"], font=font, tags="fg")

    def pulse(self):
        self.halo_on = not self.halo_on
        self.canvas.itemconfigure("halo", state="normal" if self.halo_on else "hidden")
        self.root.after(550, self.pulse)

    # ---------- interacción con el grafo ----------
    def on_press(self, e):
        v = self.node_at(e.x, e.y)
        self.drag = {"v": v, "sx": e.x, "sy": e.y, "moved": False} if v is not None else None

    def on_drag(self, e):
        d = self.drag
        if not d:
            return
        if not d["moved"] and math.hypot(e.x - d["sx"], e.y - d["sy"]) < 4:
            return
        d["moved"] = True
        W, H = self.size()
        fx = min(1.0, max(0.0, (e.x - self.PADX) / (W - 2 * self.PADX)))
        fy = min(1.0, max(0.0, (e.y - self.PADY) / (H - 2 * self.PADY)))
        self.pos[d["v"]] = (fx, fy)
        self.draw_graph()

    def on_release(self, e):
        d, self.drag = self.drag, None
        if d and not d["moved"]:
            self.pick_node(d["v"])

    def on_hover_node(self, e):
        self.canvas.configure(cursor="hand2" if self.node_at(e.x, e.y) is not None else "")

    def pick_node(self, v):
        if self.pick == "src":
            self.src, self.pick = v, "dst"
        else:
            self.dst, self.pick = v, "src"
        self.cb_src.set(str(self.src))
        self.cb_dst.set(str(self.dst))
        self.update_hint()
        self.start_run()

    # ---------- paneles ----------
    def paint(self):
        s = self.steps[self.i]
        prev = self.steps[self.i - 1] if self.i > 0 else None
        animate = self.last_painted != self.i and self.i > 0
        self.last_painted = self.i
        self.draw_graph()
        self.status.configure(text=s.msg)
        self.fill_table(s, prev, animate)
        self.queue.configure(text="   ".join(f"({d}, {v})" for d, v in s.queue) if s.queue else "La cola está vacía.")
        self.fill_log()
        final = s.kind == "end"
        if final != self.last_final:
            self.last_final = final
            self.fill_result(final)
        self.update_controls()

    def fill_table(self, s, prev, animate):
        if self.flash_job:
            self.root.after_cancel(self.flash_job)
            self.flash_job = None
        flashed = []
        for v in range(self.n):
            d = s.dist[v]
            if v == s.current:
                tag, label = "cur", "actual"
            elif v in s.visited:
                tag, label = "vis", "visitado"
            elif d != INF:
                tag, label = "que", "en cola"
            else:
                tag, label = "idle", "pendiente"
            role = " (origen)" if v == self.src and v != self.dst else " (destino)" if v == self.dst and v != self.src else \
                " (origen y destino)" if v == self.src else ""
            frm = "   " + (", ".join(map(str, s.preds[v])) if s.preds[v] else "—")
            changed = animate and prev is not None and (d != prev.dist[v] or s.preds[v] != prev.preds[v])
            self.row_tag[str(v)] = tag
            self.tree.item(str(v), values=(f"{v}{role}", fmt(d), frm, label), tags=(tag, "flash") if changed else (tag,))
            if changed:
                flashed.append(str(v))
        if flashed:
            self.flash_job = self.root.after(900, lambda ids=tuple(flashed): self.clear_flash(ids))

    def clear_flash(self, ids):
        self.flash_job = None
        for iid in ids:
            if self.tree.exists(iid):
                self.tree.item(iid, tags=(self.row_tag.get(iid, "idle"),))

    def fill_log(self):
        t = self.log
        t.configure(state="normal")
        t.delete("1.0", "end")
        for k in range(self.i + 1):
            st = self.steps[k]
            kind = "k-" + st.outcome if st.kind == "relax" else "k-" + st.kind
            t.insert("end", "• " + st.log + "\n", (kind, "now") if k == self.i else (kind,))
        t.see("end")
        t.configure(state="disabled")

    def fill_result(self, final):
        if not final:
            self.res_done.grid_remove()
            self.res_wait.grid()
            return
        self.res_wait.grid_remove()
        self.res_done.grid()
        run = self.run
        self.lb_paths.delete(0, "end")
        if not run.paths:
            self.lbl_dist_cap.configure(text=f"Camino de {self.src} a {self.dst}")
            self.lbl_dist.configure(text="Ninguno")
            self.lbl_count.configure(text="0")
            extra = (" En el grafo aleatorio las aristas siempre van de un número menor a uno mayor, "
                     "así que el destino debe ser mayor que el origen.") if self.mode.get() == "random" else ""
            self.lbl_note.configure(text="Ningún camino llega al destino desde ese origen." + extra)
            self.lb_paths.grid_remove()
            self.btn_all.grid_remove()
            return
        self.lb_paths.grid()
        self.lbl_dist_cap.configure(text=f"Distancia mínima de {self.src} a {self.dst}")
        self.lbl_dist.configure(text=str(run.distance))
        self.lbl_count.configure(text=str(run.count))
        note = "Pasa el cursor o haz clic en un camino para verlo solo en el grafo." if len(run.paths) > 1 else ""
        if run.count > len(run.paths):
            note += f" Se muestran los primeros {len(run.paths)} de {run.count}."
        self.lbl_note.configure(text=note)
        for k, pth in enumerate(run.paths):
            self.lb_paths.insert("end", f"Camino {k + 1:<3} " + " → ".join(map(str, pth)))
        self.lb_paths.configure(height=min(5, max(2, len(run.paths))))
        if len(run.paths) > 1:
            self.btn_all.grid()
        else:
            self.btn_all.grid_remove()

    def path_index_at(self, y):
        idx = self.lb_paths.nearest(y)
        if idx < 0 or idx >= self.lb_paths.size():
            return None
        bb = self.lb_paths.bbox(idx)
        return idx if bb and bb[1] <= y <= bb[1] + bb[3] else None

    def on_path_hover(self, e):
        idx = self.path_index_at(e.y)
        if idx != self.hover:
            self.hover = idx
            self.draw_graph()

    def on_path_leave(self, _e):
        if self.hover is not None:
            self.hover = None
            self.draw_graph()

    def on_path_select(self, _e):
        sel = self.lb_paths.curselection()
        self.pinned = sel[0] if sel else None
        self.draw_graph()

    def show_all_paths(self):
        self.lb_paths.selection_clear(0, "end")
        self.pinned = self.hover = None
        self.draw_graph()

    # ---------- reproductor ----------
    def update_controls(self):
        last = len(self.steps) - 1
        self._lock = True
        self.scrub.configure(to=max(1, last))
        self.scrub.set(self.i)
        self._lock = False
        self.step_lbl.configure(text=f"Paso {self.i + 1} de {last + 1}")
        at_start, at_end = self.i == 0, self.i == last
        self.b_reset.state(["disabled"] if at_start else ["!disabled"])
        self.b_prev.state(["disabled"] if at_start else ["!disabled"])
        for b in (self.b_next, self.b_vertex, self.b_end):
            b.state(["disabled"] if at_end else ["!disabled"])
        self.b_play.configure(text="⏸ Pausar" if self.playing else "▶ Reproducir")

    def go(self, k):
        self.i = max(0, min(len(self.steps) - 1, k))
        self.paint()

    def on_scrub(self, val):
        if self._lock:
            return
        k = int(round(float(val)))
        if k != self.i:
            self.stop()
            self.go(k)

    def next_vertex(self):
        for k in range(self.i + 1, len(self.steps)):
            if self.steps[k].kind in ("visit", "end"):
                self.go(k)
                return

    def delay(self):
        return {"Lenta": 1400, "Normal": 800, "Rápida": 350}.get(self.speed.get(), 800)

    def stop(self):
        self.playing = False
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        if hasattr(self, "b_play"):
            self.b_play.configure(text="▶ Reproducir")

    def toggle_play(self):
        if self.playing:
            self.stop()
            return
        if self.i >= len(self.steps) - 1:
            self.go(0)
        self.playing = True
        self.b_play.configure(text="⏸ Pausar")
        self.play_job = self.root.after(self.delay(), self.tick)

    def tick(self):
        self.play_job = None
        if not self.playing:
            return
        if self.i < len(self.steps) - 1:
            self.go(self.i + 1)
        if self.i >= len(self.steps) - 1:
            self.stop()
        else:
            self.play_job = self.root.after(self.delay(), self.tick)

    def bind_keys(self):
        def guard(fn, allow_button=True):
            def handler(e):
                w = self.root.focus_get()
                if isinstance(w, (ttk.Combobox, ttk.Spinbox, ttk.Entry, tk.Text, tk.Listbox)):
                    return
                if not allow_button and isinstance(w, ttk.Button):
                    return
                fn()
                return "break"
            return handler
        r = self.root
        r.bind("<Right>", guard(lambda: (self.stop(), self.go(self.i + 1))))
        r.bind("<Left>", guard(lambda: (self.stop(), self.go(self.i - 1))))
        r.bind("<Home>", guard(lambda: (self.stop(), self.go(0))))
        r.bind("<End>", guard(lambda: (self.stop(), self.go(len(self.steps) - 1))))
        r.bind("<space>", guard(self.toggle_play, allow_button=False))


def main():
    try:  # nitidez en pantallas de alta resolución (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
