import heapq
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# 1. KELAS GRAPH
# ============================================================
class PlayerGraph:
    def __init__(self):
        self.adj_list   = {}   # {node_id: {neighbor_id: similarity}}
        self.nodes_info = {}   # {node_id: {name, market_value, position, attributes}}

    # ── Node & Edge ──────────────────────────────────────────
    def add_node(self, nid, name, market_value, position, attributes):
        self.nodes_info[nid] = dict(
            name=name, market_value=market_value,
            position=position, attributes=attributes
        )
        self.adj_list.setdefault(nid, {})

    def add_edge(self, u, v, sim):
        self.adj_list.setdefault(u, {})[v] = sim
        self.adj_list.setdefault(v, {})[u] = sim

    # ── Metric kemiripan ─────────────────────────────────────
    def _similarity(self, a1, a2, p1, p2):
        """Euclidean terbobot + eksponensial negatif + penalti posisi."""
        weights = np.array([1.0, 1.2, 1.0, 1.1, 0.8, 0.9])
        diff     = np.abs(a1 - a2) / 100.0
        dist     = np.sqrt(np.sum((diff * weights) ** 2))
        max_dist = np.sqrt(np.sum(weights ** 2))
        base     = np.exp(-(dist / max_dist) * 5.0)

        if p1 == p2:
            factor = 1.0
        elif any(p1 in g and p2 in g for g in
                 [{'ST','CF'}, {'CAM','CM'}, {'LW','RW'}, {'CDM','CM'}]):
            factor = 0.9
        else:
            factor = 0.7
        return float(base * factor)

    # ── Bangun graf sparse ────────────────────────────────────
    def build(self, df, threshold: float = 0.20):
        """
        Bangun graf sparse: hanya tambahkan edge jika similarity ≥ threshold.
        Threshold default 0.20 agar Dijkstra benar-benar perlu 'merambat'
        lewat jalur, bukan selalu menemukan edge langsung.
        """
        self.adj_list.clear()
        self.nodes_info.clear()
        cols = ['Pace','Shooting','Passing','Dribbling','Defending','Physical']
        mat  = df[cols].values.astype(float)
        pos  = df['Position'].values

        for i in range(len(df)):
            self.add_node(i, df.iloc[i]['Name'], df.iloc[i]['MarketValue'],
                          df.iloc[i]['Position'], mat[i])

        edge_count = 0
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                sim = self._similarity(mat[i], mat[j], pos[i], pos[j])
                if sim >= threshold:
                    self.add_edge(i, j, sim)
                    edge_count += 1
        return edge_count

    # ── DIJKSTRA ──────────────────────────────────────────────
    def dijkstra(self, start_id: int, budget: float, top_n: int):
        """
        Dijkstra dari target ke semua node.
        Bobot = dissimilarity (1 − similarity).
        Jarak terkecil = jalur paling mirip (bisa transitif).

        Returns
        -------
        recommendations : list[dict]
        all_dist        : dict {node_id: total_distance}
        log             : list[str]
        """
        log = [
            "=" * 66,
            "  ALGORITMA: PENCARIAN JALUR TERPENDEK  (shortest-path berbasis dissimilarity)",
            "=" * 66,
            f"  Target : {self.nodes_info[start_id]['name']}"
            f" ({self.nodes_info[start_id]['position']})"
            f"  –  €{self.nodes_info[start_id]['market_value']}M",
            f"  Budget : ≤ €{budget}M",
            "-" * 66,
            "  Bobot edge = 1 − similarity  →  jarak kecil = sangat mirip",
            "  Jalur bisa melewati pemain PERANTARA (kemiripan transitif)",
            "=" * 66,
            "",
        ]

        # ── Inisialisasi ──
        INF  = float('inf')
        dist = {n: INF  for n in self.nodes_info}
        prev = {n: None for n in self.nodes_info}
        dist[start_id] = 0.0
        pq = [(0.0, start_id)]
        visited = set()
        steps   = []

        # ── Relaxasi ──
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            label = (f"  ✔ Kunjungi [{self.nodes_info[u]['name']:22s}]"
                     f"  dist = {d:.4f}")
            if u == start_id:
                label += "  ← START"
            steps.append(label)

            for v, sim in self.adj_list.get(u, {}).items():
                cost = 1.0 - sim          # dissimilarity sebagai jarak
                nd   = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        log += steps
        log += ["", "-" * 66]

        # ── Rekonstruksi jalur ──
        def path_of(nid):
            p, cur = [], nid
            while cur is not None:
                p.append(self.nodes_info[cur]['name'])
                cur = prev[cur]
            return list(reversed(p))

        # ── Kumpulkan kandidat dalam budget ──
        candidates = []
        for nid in self.nodes_info:
            if nid == start_id:
                continue
            mv = self.nodes_info[nid]['market_value']
            if mv <= budget and dist[nid] < INF:
                p       = path_of(nid)
                dir_sim = self.adj_list.get(start_id, {}).get(nid, 0.0)
                # Similarity transitif: konversi jarak kembali ke 0-1
                t_sim   = float(np.exp(-dist[nid]))
                candidates.append(dict(
                    node_id       = nid,
                    name          = self.nodes_info[nid]['name'],
                    position      = self.nodes_info[nid]['position'],
                    market_value  = mv,
                    direct_sim    = dir_sim,      # similarity edge langsung
                    trans_sim     = t_sim,        # similarity via jalur terpendek
                    dijkstra_dist = dist[nid],
                    path          = p,
                    hops          = len(p) - 1,   # jumlah lompatan
                    attributes    = self.nodes_info[nid]['attributes'],
                ))

        # Urut: jarak terpendek dulu, lalu harga termurah
        candidates.sort(key=lambda x: (x['dijkstra_dist'], x['market_value']))
        recommendations = candidates[:top_n]

        log.append(f"  TOP {len(recommendations)} REKOMENDASI:")
        log.append("")
        for i, r in enumerate(recommendations):
            log.append(f"  {i+1:2d}. {r['name']:22s}"
                       f"  dist={r['dijkstra_dist']:.4f}"
                       f"  sim-trans={r['trans_sim']:.1%}"
                       f"  hops={r['hops']}")
            log.append(f"      jalur: {' → '.join(r['path'])}")
        log.append("=" * 66)

        return recommendations, dist, log

    def direct_sim(self, u, v):
        return self.adj_list.get(u, {}).get(v, 0.0)


# ============================================================
# 2. DATASET DEFAULT (35 PEMAIN)
# ============================================================
def default_data():
    return pd.DataFrame({
        'Name': [
            "Erling Haaland","Kylian Mbappé","Vinícius Júnior","Jude Bellingham",
            "Bukayo Saka","Rodri","Phil Foden","Jamal Musiala","Florian Wirtz",
            "Lautaro Martínez","Victor Osimhen","Rafael Leão","Khvicha Kvaratskhelia",
            "Alexander Isak","Martin Ødegaard","Declan Rice","Eduardo Camavinga",
            "Pedri","Gavi","Rodrygo","Lois Openda","Donyell Malen","Jeremy Doku",
            "Brajan Gruda","Johan Bakayoko","João Palhinha","Xavi Simons",
            "Maximilian Beier","Matt O'Riley","Santiago Giménez",
            "Serhou Guirassy","Viktor Gyökeres","Mohamed Amoura",
            "Crysencio Summerville","Georges Mikautadze",
        ],
        'Position': [
            "ST","ST","LW","CAM","RW","CDM","CAM","CAM","CAM","ST",
            "ST","LW","LW","ST","CAM","CDM","CM","CM","CM","RW",
            "ST","ST","LW","CAM","RW","CDM","CAM","ST","CAM","ST",
            "ST","ST","LW","LW","ST",
        ],
        'MarketValue': [
            180,180,150,140,130,110,100,95,90,85,
            65,60,55,50,50,50,50,45,45,40,
            35,30,25,20,25,30,28,22,15,18,
            12,10,8,10,7,
        ],
        'Pace':      [89,97,95,82,87,60,85,87,84,83,92,94,91,88,79,72,84,82,80,89,91,93,94,80,88,65,86,89,78,82,85,87,90,89,81],
        'Shooting':  [94,91,85,87,83,75,85,82,80,88,89,84,83,86,82,65,68,70,65,82,85,84,78,78,76,60,80,82,75,85,88,83,76,78,82],
        'Passing':   [68,84,82,88,85,90,89,86,92,75,70,80,84,78,91,82,80,90,84,80,72,78,75,82,80,72,86,74,88,68,65,72,74,76,70],
        'Dribbling': [78,92,94,86,89,80,91,93,90,84,79,90,93,86,89,78,85,91,86,90,80,87,92,84,87,70,88,83,82,78,77,80,85,88,79],
        'Defending': [50,38,32,70,60,88,55,50,45,40,45,30,35,40,65,85,75,70,72,35,42,35,30,55,45,87,48,40,55,38,48,42,28,32,35],
        'Physical':  [92,78,68,82,72,86,65,60,58,80,85,78,72,75,60,84,78,62,68,65,80,72,65,70,68,88,62,72,65,78,85,88,62,65,72],
    })


# ============================================================
# 3. VISUALISASI
# ============================================================
def draw_main_graph(graph, target_id, recs):
    """
    Graf utama: target di tengah, rekomendasi di keliling.
    Warna edge: merah = jalur terpendek (mungkin via perantara),
    abu = edge langsung lain yang ada di graf sparse.
    """
    G = nx.Graph()

    # Tambahkan semua node rekomendasi + perantara di jalur mereka
    shown = {target_id}
    path_edges = set()
    for r in recs:
        path = r['path']
        for name in path:
            nid = next(i for i, info in graph.nodes_info.items()
                       if info['name'] == name)
            shown.add(nid)
        # Edges di jalur
        for k in range(len(path) - 1):
            u_id = next(i for i, info in graph.nodes_info.items()
                        if info['name'] == path[k])
            v_id = next(i for i, info in graph.nodes_info.items()
                        if info['name'] == path[k+1])
            path_edges.add((min(u_id,v_id), max(u_id,v_id)))

    for nid in shown:
        G.add_node(nid)

    # Semua edge di antara node yang ditampilkan
    for u in shown:
        for v, sim in graph.adj_list.get(u, {}).items():
            if v in shown and u < v:
                G.add_edge(u, v, similarity=sim,
                           is_path=(min(u,v), max(u,v)) in path_edges)

    pos = nx.spring_layout(G, seed=42, k=2.8)

    fig, ax = plt.subplots(figsize=(15, 10))

    rec_ids       = {r['node_id'] for r in recs}
    node_colors   = ['#FF4444' if n == target_id
                     else '#44BB88' if n in rec_ids
                     else '#AACCFF' for n in G.nodes()]
    node_sizes    = [3800 if n == target_id
                     else 2400 if n in rec_ids
                     else 1400 for n in G.nodes()]

    # Edge jalur terpendek (merah tebal)
    path_edge_list = [(u,v) for u,v,d in G.edges(data=True) if d.get('is_path')]
    other_edge_list= [(u,v) for u,v,d in G.edges(data=True) if not d.get('is_path')]

    nx.draw_networkx_edges(G, pos, edgelist=other_edge_list,
                           width=1.2, edge_color='#CCCCCC', alpha=0.5, ax=ax)
    for u, v in path_edge_list:
        sim = G[u][v]['similarity']
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                               width=2 + 8*sim, edge_color='#FF4444',
                               alpha=0.85, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='black', linewidths=2, ax=ax)

    labels = {n: graph.nodes_info[n]['name'].split()[-1] +
                 f"\n€{graph.nodes_info[n]['market_value']}M"
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

    # Label similarity hanya di jalur
    edge_labels = {(u,v): f"{G[u][v]['similarity']:.1%}"
                   for u,v in path_edge_list}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax,
                                 bbox=dict(facecolor='white', alpha=0.75,
                                           edgecolor='#FF4444', boxstyle='round'))

    legend_handles = [
        mpatches.Patch(facecolor='#FF4444', edgecolor='black', label='🎯 Target'),
        mpatches.Patch(facecolor='#44BB88', edgecolor='black', label='✅ Rekomendasi'),
        mpatches.Patch(facecolor='#AACCFF', edgecolor='black', label='🔗 Perantara (node di jalur)'),
        mpatches.Patch(facecolor='#FF4444', alpha=0.7, edgecolor='none',
                       label='━ Jalur Terpendek'),
        mpatches.Patch(facecolor='#CCCCCC', alpha=0.7, edgecolor='none',
                       label='─ Edge sparse lain'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.01, 1), fontsize=10, framealpha=0.9)
    ax.set_title("Graf Kemiripan — Jalur Terpendek (dissimilarity terkecil)\n"
                 "Garis merah tebal = jalur terpendek dari target ke rekomendasi",
                 fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_comparison_bar(tname, tattrs, rname, rattrs):
    labels = ['Pace','Shooting','Passing','Dribbling','Defending','Physical']
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, tattrs, w, label=tname, color='#FF4444', alpha=0.85)
    ax.bar(x + w/2, rattrs, w, label=rname, color='#44BB88', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Nilai", fontsize=11)
    ax.set_title(f"Perbandingan: {tname} vs {rname}",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                f'{int(h)}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# 4. STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="DSS Alternatif Pemain",
    layout="wide", page_icon="⚽"
)

# ── Header ────────────────────────────────────────────────────────────────
st.title("⚽ Decision Support System — Pencari Pemain Alternatif")

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Konfigurasi")

uploaded = st.sidebar.file_uploader("📁 Upload CSV (opsional)", type=['csv'])
if uploaded:
    try:
        df_up = pd.read_csv(uploaded)
        req   = ['Name','Position','MarketValue',
                 'Pace','Shooting','Passing','Dribbling','Defending','Physical']
        if all(c in df_up.columns for c in req):
            st.session_state.df   = df_up
            st.session_state.done = False
            st.sidebar.success("✅ Data berhasil diunggah!")
        else:
            st.sidebar.error(f"❌ Kolom wajib: {', '.join(req)}")
    except Exception as e:
        st.sidebar.error(f"❌ Gagal membaca: {e}")

if 'df' not in st.session_state:
    st.session_state.df   = default_data()
    st.session_state.done = False

df = st.session_state.df

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Parameter")
target_name = st.sidebar.selectbox("Pemain Target", df['Name'].tolist())
budget      = st.sidebar.slider("Budget Maks (Juta €)", 0,
                                int(df['MarketValue'].max()), 50, 5)
n_rec       = st.sidebar.slider("Jumlah Rekomendasi", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Tingkat Kemiripan Minimal")

THRESHOLD_OPTIONS = {
    "🟢 Longgar — semua pemain terhubung":        0.05,
    "🟡 Sedang — pemain cukup mirip terhubung":   0.20,
    "🟠 Ketat — hanya pemain mirip yang terhubung": 0.35,
    "🔴 Sangat Ketat — hanya yang sangat mirip":  0.50,
}
threshold_label = st.sidebar.radio(
    "Seberapa mirip dua pemain harus agar terhubung di graf?",
    list(THRESHOLD_OPTIONS.keys()),
    index=1,                          # default: Sedang
)
threshold = THRESHOLD_OPTIONS[threshold_label]

THRESHOLD_DESC = {
    "🟢 Longgar — semua pemain terhubung":
        "Hampir semua pemain saling terhubung langsung. "
        "Algoritma mudah menemukan jalur, tapi hasilnya kurang selektif.",
    "🟡 Sedang — pemain cukup mirip terhubung":
        "Keseimbangan antara keterhubungan dan selektivitas. "
        "Algoritma mulai mencari jalur lewat perantara. **(Disarankan)**",
    "🟠 Ketat — hanya pemain mirip yang terhubung":
        "Graf lebih jarang. Algoritma wajib melewati perantara untuk menemukan alternatif. "
        "Hasil lebih selektif, tapi ada risiko beberapa pemain tidak terjangkau.",
    "🔴 Sangat Ketat — hanya yang sangat mirip":
        "Graf sangat jarang. Beberapa pemain mungkin terisolasi (jarak = ∞). "
        "Gunakan jika ingin rekomendasi yang benar-benar mirip saja.",
}
st.sidebar.caption(THRESHOLD_DESC[threshold_label])

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔍 Cari Pemain Alternatif", use_container_width=True)

# ── Proses ────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Membangun graf sparse & menjalankan pencarian jalur..."):
        g = PlayerGraph()
        n_edges   = g.build(df, threshold=threshold)
        target_id = int(df[df['Name'] == target_name].index[0])
        recs, all_dist, log = g.dijkstra(target_id, budget, n_rec)

    st.session_state.graph            = g
    st.session_state.recs             = recs
    st.session_state.all_dist         = all_dist
    st.session_state.log              = log
    st.session_state.target_id        = target_id
    st.session_state.n_edges          = n_edges
    st.session_state.threshold_label  = threshold_label
    st.session_state.done             = True

    if recs:
        st.success(f"✅ Pencarian selesai — {len(recs)} rekomendasi dalam budget €{budget}M!")
    else:
        st.warning("⚠️ Tidak ada pemain dalam budget atau threshold terlalu tinggi (graf tidak terhubung). "
                   "Coba turunkan threshold atau naikkan budget.")

# ── Dataset ───────────────────────────────────────────────────────────────
with st.expander("📋 Dataset Pemain", expanded=False):
    st.dataframe(df, use_container_width=True)

# ── Hasil ─────────────────────────────────────────────────────────────────
if st.session_state.get('done', False):
    g         = st.session_state.graph
    recs      = st.session_state.recs
    all_dist  = st.session_state.all_dist
    log       = st.session_state.log
    target_id = st.session_state.target_id
    n_edges         = st.session_state.n_edges
    threshold_label = st.session_state.get('threshold_label', '')
    t_info    = g.nodes_info[target_id]

    # ── Info grafik & target ──
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Target",        t_info['name'])
    c2.metric("📍 Posisi",        t_info['position'])
    c3.metric("💶 Nilai Pasar",   f"€{t_info['market_value']}M")
    c4.metric("💰 Budget",        f"≤ €{budget}M")
    level_short = threshold_label.split("—")[0].strip() if "—" in threshold_label else threshold_label
    c5.metric("🔗 Koneksi Graf", f"{n_edges} edge",
              help=f"Tingkat kemiripan: {threshold_label}")

    if not recs:
        st.error("Tidak ada rekomendasi. Coba pilih tingkat kemiripan yang lebih **Longgar** "
                 "di sidebar, atau naikkan budget.")
        st.stop()

    # ── Tabel hasil ──
    st.subheader("✅ Hasil Rekomendasi")
    rows = []
    for i, r in enumerate(recs):
        sim = r['direct_sim']
        level = ("🟢 Sangat Mirip" if sim >= 0.8 else
                 "🟡 Mirip"         if sim >= 0.7 else
                 "🟠 Cukup Mirip"   if sim >= 0.6 else "🔴 Kurang Mirip")
        rows.append({
            'Rank'               : i + 1,
            'Nama'               : r['name'],
            'Posisi'             : r['position'],
            'Harga (€M)'         : r['market_value'],
            'Direct Similarity'  : f"{r['direct_sim']:.1%}",
            'Trans. Similarity'  : f"{r['trans_sim']:.1%}",
            'Jarak Terpendek'     : f"{r['dijkstra_dist']:.4f}",
            'Hops'               : r['hops'],
            'Level'              : level,
            'Hemat'              : f"€{t_info['market_value'] - r['market_value']}M",
            'Jalur'              : ' → '.join(r['path']),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)

    # ── Metrics ringkasan ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rekomendasi ditemukan", len(recs))
    avg_d = float(np.mean([r['dijkstra_dist'] for r in recs]))
    m2.metric("Rata-rata jarak", f"{avg_d:.4f}")
    avg_s = float(np.mean([r['trans_sim'] for r in recs]))
    m3.metric("Rata-rata sim. transitif", f"{avg_s:.1%}")
    multi_hop = sum(1 for r in recs if r['hops'] > 1)
    m4.metric("Via perantara (hops > 1)", multi_hop,
              help="Rekomendasi yang ditemukan LEWAT pemain perantara (khas analisis graf)")

    st.markdown("---")

    # ── Graf utama ──
    st.subheader("🗺️ Graf Kemiripan & Jalur Terpendek")
    st.pyplot(draw_main_graph(g, target_id, recs))
    st.caption(
        "🔴 Target  |  🟢 Rekomendasi  |  🔵 Perantara  |  "
        "**Garis merah** = jalur terpendek  |  Abu = edge sparse lain"
    )

    st.markdown("---")

    # ── Detail per rekomendasi ──
    st.subheader("🔎 Detail Rekomendasi")
    sel_name = st.selectbox("Pilih rekomendasi untuk inspeksi:",
                            [r['name'] for r in recs])
    sel = next(r for r in recs if r['name'] == sel_name)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Nama",                sel['name'])
    d2.metric("Harga",               f"€{sel['market_value']}M")
    d3.metric("Direct Similarity",   f"{sel['direct_sim']:.1%}")
    d4.metric("Trans. Similarity",   f"{sel['trans_sim']:.1%}",
              help="exp(−dijkstra_dist) — similarity melalui jalur terpendek")

    # Bar atribut & tabel selisih
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.markdown("##### 📈 Perbandingan Atribut")
        st.pyplot(draw_comparison_bar(
            target_name, g.nodes_info[target_id]['attributes'],
            sel_name,    sel['attributes']
        ))
    with col_table:
        st.markdown("##### 📋 Selisih Atribut")
        attr_labels  = ['Pace','Shooting','Passing','Dribbling','Defending','Physical']
        tattrs       = g.nodes_info[target_id]['attributes'].astype(int)
        rattrs       = sel['attributes'].astype(int)
        diff         = np.abs(tattrs - rattrs)
        diff_df      = pd.DataFrame({
            'Atribut'  : attr_labels,
            target_name: tattrs,
            sel_name   : rattrs,
            'Selisih'  : diff,
        })
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

        saving     = t_info['market_value'] - sel['market_value']
        saving_pct = saving / t_info['market_value'] if t_info['market_value'] > 0 else 0
        st.metric("💰 Potensi Hemat", f"€{saving}M", delta=f"{saving_pct:.0%}")

    # ── Log ──
    with st.expander("🔧 Log Perhitungan Jalur"):
        st.text('\n'.join(log))

        st.markdown("### Semua Pemain — Jarak Terpendek dari Target")
        all_rows = []
        for nid, d in sorted(all_dist.items(), key=lambda x: x[1]):
            if nid == target_id:
                continue
            info = g.nodes_info[nid]
            all_rows.append({
                'Nama'               : info['name'],
                'Posisi'             : info['position'],
                'Jarak Terpendek'     : f"{d:.4f}" if d < float('inf') else "∞ (tidak terhubung)",
                'Trans. Similarity'  : f"{np.exp(-d):.1%}" if d < float('inf') else "—",
                'Direct Sim.'        : f"{g.direct_sim(target_id, nid):.1%}",
                'Harga (€M)'         : info['market_value'],
                'Budget'             : '✅' if info['market_value'] <= budget else '❌',
            })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)