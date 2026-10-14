import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import httpx
from bs4 import BeautifulSoup
import urllib.parse
import time
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union

# ==========================================
# 1. KONFIGURACJA STRONY STREAMLIT & STYLE
# ==========================================
st.set_page_config(
    page_title="Site Focus Score & Site Radius Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla nowoczesnego wyglądu UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .filter-summary {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #166534;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .cf-warning-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #991B1B;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .cta-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        color: #FFFFFF;
        text-align: center;
        margin-top: 3.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .cta-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.8rem;
    }
    .cta-desc {
        font-size: 1.05rem;
        color: #CBD5E1;
        max-width: 720px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }
    .cta-btn {
        display: inline-block;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: #FFFFFF !important;
        padding: 0.8rem 2.2rem;
        border-radius: 8px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .cta-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Stałe aplikacji
MAX_URL_LIMIT = 150
COOLDOWN_SECONDS = 45

# Realistyczne nagłówki nowoczesnej przeglądarki (omijanie podstawowych blokad antybotowych / Cloudflare)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1"
}

# Rozszerzenia plików niebędących stronami HTML
NON_HTML_EXTENSIONS = (
    # Obrazy i grafika
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico", ".tiff", ".avif",
    # Dokumenty i archiwa
    ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".txt", ".xml",
    # Multimedia
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".webm", ".mkv",
    # Czcionki
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Skrypty i arkusze stylów
    ".js", ".css", ".json", ".map", ".rss", ".atom"
)

# ==========================================
# 2. DARMOWY MODEL LOKALNY (ZERO API)
# ==========================================
@st.cache_resource(show_spinner="⏳ Ładowanie lokalnego modelu wektorowego (intfloat/multilingual-e5-small)...")
def load_embedding_model():
    """
    Ładuje lokalny model multilingual-e5-small z HuggingFace/sentence-transformers.
    Cache'owany w RAM, ładuje się tylko raz przy starcie aplikacji.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    return model

# ==========================================
# 3. ZABEZPIECZENIA: COOLDOWN & ANTYSPAM
# ==========================================
def check_cooldown() -> Tuple[bool, int]:
    """Sprawdza sesyjny cooldown (45 sekund) pomiędzy kolejnymi analizami."""
    now = time.time()
    last_run = st.session_state.get("last_run", 0)
    elapsed = now - last_run
    if elapsed < COOLDOWN_SECONDS:
        return False, int(COOLDOWN_SECONDS - elapsed)
    return True, 0

def update_cooldown():
    """Aktualizuje znacznik czasu ostatniej analizy."""
    st.session_state["last_run"] = time.time()

# ==========================================
# 4. FILTRACJA HTML & PARSOWANIE URL
# ==========================================
def extract_clean_path(url: str) -> str:
    """Konwertuje ścieżkę URL do czytelnej postaci semantycznej."""
    try:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return "strona główna (home)"
        decoded = urllib.parse.unquote(path)
        clean = decoded.replace("-", " ").replace("_", " ").replace("/", " > ")
        return clean
    except Exception:
        return "podstrona"

def is_valid_html_url(url: str, content_type: Optional[str] = None, http_status: Optional[any] = None, ignore_query: bool = True) -> bool:
    """
    Sprawdza, czy dany adres URL reprezentuje podstronę HTML:
    - Odrzuca obrazy, skrypty, pliki binarne, dokumenty PDF itp.
    - Odrzuca błędy 404, 403 oraz przekierowania 301/302 (jeśli podano HTTP_STATUS).
    - Sprawdza CONTENT_TYPE (wymaga text/html lub odmian HTML).
    - Opcjonalnie ignoruje adresy z parametrami zapytania '?'.
    """
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return False

    # Ignoruj URL ze znakiem zapytania '?' (np. parametry śledzące, duplikaty)
    if ignore_query and "?" in url:
        return False

    parsed = urllib.parse.urlparse(url)
    path_lower = parsed.path.lower()

    # Odrzucenie po rozszerzeniu pliku
    for ext in NON_HTML_EXTENSIONS:
        if path_lower.endswith(ext):
            return False

    # Sprawdzenie kodu HTTP status (tylko 200)
    if http_status is not None and pd.notna(http_status):
        try:
            status_val = int(float(http_status))
            if status_val != 200:
                return False
        except (ValueError, TypeError):
            pass

    # Sprawdzenie Content-Type
    if content_type is not None and pd.notna(content_type):
        ct = str(content_type).lower().strip()
        # Wyklucz jawnie nie-HTML
        if any(non in ct for non in ["image/", "application/pdf", "text/plain", "text/css", "javascript", "audio/", "video/"]):
            return False
        # Jeśli pole nie jest puste, musi zawierać 'html'
        if "html" not in ct:
            return False

    return True

# ==========================================
# 5. POBIERANIE SITEMAP & OCHRONA 403/CLOUDFLARE
# ==========================================
def fetch_remote_content(url: str, timeout: float = 12.0) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Pobiera treść URL z realistycznymi nagłówkami przeglądarkowymi.
    Zwraca (content, error_code).
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, verify=False, headers=BROWSER_HEADERS) as client:
            resp = client.get(url)
            if resp.status_code == 403:
                return None, "CLOUDFLARE_403"
            elif resp.status_code == 429:
                return None, "RATE_LIMIT_429"
            elif resp.status_code >= 400:
                return None, f"HTTP_{resp.status_code}"
            return resp.content, None
    except httpx.ConnectError:
        return None, "CONNECT_ERROR"
    except httpx.TimeoutException:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"ERROR_{type(e).__name__}"

def inspect_and_parse_sitemap(xml_content: bytes) -> Dict[str, Union[str, List[str]]]:
    """
    Sprawdza, czy sitemapa to Sitemap Index (<sitemapindex>) czy standardowa mapa (<urlset>).
    Zwraca:
    {
        "type": "index" | "urlset" | "empty",
        "sub_sitemaps": [url1, url2, ...],
        "urls": [url1, url2, ...]
    }
    """
    try:
        soup = BeautifulSoup(xml_content, "xml")
    except Exception:
        soup = BeautifulSoup(xml_content, "html.parser")

    # Sprawdzenie czy to Sitemap Index
    sitemaps = soup.find_all("sitemap")
    if sitemaps:
        sub_urls = []
        for s in sitemaps:
            loc = s.find("loc")
            if loc and loc.get_text(strip=True):
                sub_urls.append(loc.get_text(strip=True))
        if sub_urls:
            return {"type": "index", "sub_sitemaps": sub_urls, "urls": []}

    # Sprawdzenie czy to <urlset>
    urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    if urls:
        return {"type": "urlset", "sub_sitemaps": [], "urls": urls}

    # Fallback tekstowy (lista URL w pliku tekstowym)
    text = xml_content.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("http")]
    if lines:
        return {"type": "urlset", "sub_sitemaps": [], "urls": lines}

    return {"type": "empty", "sub_sitemaps": [], "urls": []}

def scrape_single_url(client: httpx.Client, url: str) -> Dict[str, str]:
    """Pobiera pojedynczą podstronę i wyciąga Title, Meta Description oraz pierwszy H1."""
    try:
        resp = client.get(url, headers=BROWSER_HEADERS, follow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Title
            title = soup.title.get_text(strip=True) if soup.title else ""
            
            # Meta Description
            desc_tag = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "description"})
            if not desc_tag:
                desc_tag = soup.find("meta", attrs={"property": lambda x: x and x.lower() == "og:description"})
            description = desc_tag.get("content", "").strip() if desc_tag and desc_tag.get("content") else ""
            
            # H1
            h1_tag = soup.find("h1")
            h1 = h1_tag.get_text(strip=True) if h1_tag else ""
            
            return {"url": url, "title": title, "description": description, "h1": h1, "status": "OK"}
        elif resp.status_code == 403:
            return {"url": url, "title": "", "description": "", "h1": "", "status": "Blokada 403 / Cloudflare"}
        else:
            return {"url": url, "title": "", "description": "", "h1": "", "status": f"HTTP {resp.status_code}"}
    except Exception:
        return {"url": url, "title": "", "description": "", "h1": "", "status": "Błąd połączenia"}

def run_scraper(urls: List[str], progress_bar, status_text) -> List[Dict[str, str]]:
    """Wielowątkowy scraper z timeoutem 5s na URL i realistycznym User-Agentem."""
    results = []
    total = len(urls)
    with httpx.Client(timeout=5.0, verify=False) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {executor.submit(scrape_single_url, client, u): u for u in urls}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_url):
                res = future.result()
                results.append(res)
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Pobrano {completed} z {total} adresów URL...")
    # Zachowaj pierwotną kolejność
    url_order = {u: i for i, u in enumerate(urls)}
    results.sort(key=lambda x: url_order.get(x["url"], 9999))
    return results

# ==========================================
# 6. MATEMATYKA & WEKTORYZACJA (CACHE_DATA)
# ==========================================
@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def compute_semantic_analysis(records: Tuple[Tuple[str, str, str, str], ...]) -> Dict:
    """
    Główna funkcja wyliczająca Site Focus Score i Site Radius.
    Argument `records` to krotka: (url, title, description, h1).
    Cache'owana w Streamlit na 1 godzinę.
    """
    model = load_embedding_model()

    urls = [r[0] for r in records]
    titles = [r[1] for r in records]
    descriptions = [r[2] for r in records]
    h1s = [r[3] for r in records]

    # Przygotowanie tekstów z prefiksem 'passage: ' (zgodnie z wymaganiem modeli E5)
    texts = []
    clean_paths = []
    for u, t, desc, h1 in zip(urls, titles, descriptions, h1s):
        c_path = extract_clean_path(u)
        clean_paths.append(c_path)
        
        parts = [f"Ścieżka URL: {c_path}"]
        if t:
            parts.append(f"Tytuł: {t}")
        if desc:
            parts.append(f"Opis: {desc}")
        if h1:
            parts.append(f"H1: {h1}")
            
        passage = "passage: " + " | ".join(parts)
        texts.append(passage)

    # Obliczenie wektorów z normalize_embeddings=True (norma L2 = 1)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Centroid: średnia arytmetyczna wektorów, znormalizowana normą L2
    raw_centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(raw_centroid)
    centroid = raw_centroid / (norm if norm > 0 else 1.0)

    # Cosine similarity i Page Radius dla każdej podstrony
    cosine_sim = np.dot(embeddings, centroid)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    page_radius = 1.0 - cosine_sim

    # Metryki serwisu
    site_radius = float(np.mean(page_radius))
    p90_radius = float(np.percentile(page_radius, 90))

    # Skalowanie Site Focus Score dla multilingual-e5-small [0.65, 0.92] -> [0%, 100%]
    raw_sim = 1.0 - site_radius
    site_focus_score = float(np.clip((raw_sim - 0.65) / (0.92 - 0.65) * 100.0, 0.0, 100.0))

    # Klasyfikacja podstron (percentyle i odchylenie standardowe)
    p70 = float(np.percentile(page_radius, 70))
    p88 = float(np.percentile(page_radius, 88))
    std_thresh = float(np.mean(page_radius) + 1.2 * np.std(page_radius))
    drift_thresh = min(p88, std_thresh) if len(page_radius) >= 10 else p88

    statuses = []
    status_colors = []
    for r in page_radius:
        if r <= p70:
            statuses.append("Rdzeń tematyczny (Core)")
            status_colors.append("#10B981") # Zielony
        elif r <= drift_thresh:
            statuses.append("Temat poboczny (Supporting)")
            status_colors.append("#F59E0B") # Żółty/Bursztynowy
        else:
            statuses.append("Ryzyko Topical Drift / Content Pruning")
            status_colors.append("#EF4444") # Czerwony

    # Redukcja wymiarów z 384D do 2D za pomocą PCA
    all_vectors = np.vstack([embeddings, centroid.reshape(1, -1)])
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(all_vectors)

    page_coords = coords_2d[:-1]
    centroid_coord = coords_2d[-1]

    return {
        "urls": urls,
        "titles": titles,
        "descriptions": descriptions,
        "h1s": h1s,
        "clean_paths": clean_paths,
        "embeddings": embeddings,
        "centroid": centroid,
        "cosine_sim": cosine_sim,
        "page_radius": page_radius,
        "site_radius": site_radius,
        "p90_radius": p90_radius,
        "site_focus_score": site_focus_score,
        "statuses": statuses,
        "status_colors": status_colors,
        "pca_x": page_coords[:, 0],
        "pca_y": page_coords[:, 1],
        "centroid_pca_x": float(centroid_coord[0]),
        "centroid_pca_y": float(centroid_coord[1]),
    }

# ==========================================
# 7. SIDEBAR: INFORMACJE & PARAMETRY
# ==========================================
with st.sidebar:
    st.markdown("### 🎯 Site Focus Analyzer")
    st.caption("Semantyczna analiza spójności tematycznej witryny")
    st.markdown("---")
    
    st.markdown("#### 💡 Dwa sposoby działania:")
    st.markdown("""
    1. **Audyt z Clusteric**:
       Pełna analiza na podstawie wgranego pliku CSV/XLSX. Narzędzie automatycznie wyciąga `URL`, `TITLE`, `DESCRIPTION` oraz `H1`, odrzucając zdjęcia, pliki, skrypty oraz przekierowania.
    2. **Mapa witryny (Sitemap.xml)**:
       Uproszczona analiza z sitemapy z automatyczną obsługą **Sitemap Index** (wybór pod-sitemap: posty, strony itp.) oraz odpornością na Cloudflare.
    """)
    
    st.markdown("#### ⚙️ Bezpieczeństwo i wydajność:")
    st.markdown("""
    - **Model NLP**: `multilingual-e5-small` (100% lokalny)
    - **Filtr zasobów**: automatyczne pomijanie zdjęć (`.jpg`, `.png`), skryptów i plików
    - **Hard Cap**: Max **150 URL** na pojedynczy audyt
    - **Rate Limit**: Cooldown **45 sekund**
    - **Cache**: Pamięć podręczna na 1 godzinę
    """)
    
    st.markdown("---")
    st.markdown("Stworzone przez: **Jan Wójcicki**")
    st.markdown("[janwojcicki.pl](https://janwojcicki.pl) | [LinkedIn](https://www.linkedin.com/in/jan-wojcicki)")

# ==========================================
# 8. NAGŁÓWEK GŁÓWNY APLIKACJI
# ==========================================
st.markdown('<div class="main-header">🎯 Site Focus Score & Site Radius Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Zbadaj spójność tematyczną (Topical Authority) swojej witryny, wyznacz semantyczny Centroid '
    'oraz zidentyfikuj podstrony rozmywające profil domeny kwalifikujące się do Content Pruningu.</div>',
    unsafe_allow_html=True
)

# ==========================================
# 9. DWA TRYBY DANYCH WEJŚCIOWYCH
# ==========================================
tab_clusteric, tab_sitemap = st.tabs([
    "📊 Tryb 1: Audyt z Clusteric (CSV / XLSX)", 
    "🗺️ Tryb 2: Uproszczona analiza z Mapy Witryny (Sitemap.xml)"
])

prepared_records: Optional[List[Tuple[str, str, str, str]]] = None

# --- TRYB 1: CLUSTERIC AUDIT ---
with tab_clusteric:
    st.markdown("##### Wgraj plik audytu z Clusteric")
    st.caption("Narzędzie automatycznie wyekstrahuje kolumny **URL, TITLE, DESCRIPTION i H1** oraz odfiltruje obrazy, skrypty, błędy i przekierowania.")
    
    clusteric_file = st.file_uploader(
        "Wybierz plik audytu Clusteric (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="uploader_clusteric",
        help="Obsługuje bezpośrednie eksporty z programu Clusteric Search Auditor."
    )
    
    if clusteric_file is not None:
        try:
            # Wczytanie pliku z obsługą .csv, .xlsx, .xls oraz formatów hybrydowych
            file_name_lower = clusteric_file.name.lower()
            if file_name_lower.endswith(".csv"):
                try:
                    df_raw = pd.read_csv(clusteric_file, sep=None, engine="python", encoding="utf-8")
                except Exception:
                    clusteric_file.seek(0)
                    df_raw = pd.read_csv(clusteric_file, sep=None, engine="python", encoding="cp1250")
            else:
                try:
                    # Standardowy odczyt Excel (openpyxl dla .xlsx, xlrd dla .xls)
                    df_raw = pd.read_excel(clusteric_file)
                except Exception as excel_err:
                    clusteric_file.seek(0)
                    # Fallback 1: plik .xls może być plikiem tekstowym CSV/TSV
                    try:
                        df_raw = pd.read_csv(clusteric_file, sep=None, engine="python", encoding="utf-8")
                    except Exception:
                        clusteric_file.seek(0)
                        try:
                            df_raw = pd.read_csv(clusteric_file, sep=None, engine="python", encoding="cp1250")
                        except Exception:
                            # Fallback 2: plik .xls może być tabelą HTML zapisaną jako .xls
                            clusteric_file.seek(0)
                            try:
                                tables = pd.read_html(clusteric_file)
                                if tables:
                                    df_raw = tables[0]
                                else:
                                    raise excel_err
                            except Exception:
                                raise excel_err

            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            cols = list(df_raw.columns)
            
            def match_column(candidates: List[str], available: List[str]) -> Optional[str]:
                for cand in candidates:
                    for col in available:
                        if cand.lower() == col.lower():
                            return col
                for cand in candidates:
                    for col in available:
                        if cand.lower() in col.lower():
                            return col
                return None

            col_url_guess = match_column(["URL", "Address", "Adres"], cols) or cols[0]
            col_title_guess = match_column(["TITLE", "Title 1", "Tytuł"], cols)
            col_desc_guess = match_column(["DESCRIPTION", "Meta Description", "Opis"], cols)
            col_h1_guess = match_column(["H1", "H1-1", "Nagłówek 1"], cols)
            col_ct_guess = match_column(["CONTENT_TYPE", "Content Type"], cols)
            col_status_guess = match_column(["HTTP_STATUS", "Status Code", "Status"], cols)

            with st.expander("⚙️ Wykryte kolumny audytu (opcjonalna weryfikacja)", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    col_url = st.selectbox("Kolumna URL (*)", cols, index=cols.index(col_url_guess) if col_url_guess in cols else 0)
                with c2:
                    col_title = st.selectbox("Kolumna TITLE", ["(Brak)"] + cols, index=(cols.index(col_title_guess) + 1) if col_title_guess in cols else 0)
                with c3:
                    col_desc = st.selectbox("Kolumna DESCRIPTION", ["(Brak)"] + cols, index=(cols.index(col_desc_guess) + 1) if col_desc_guess in cols else 0)
                with c4:
                    col_h1 = st.selectbox("Kolumna H1", ["(Brak)"] + cols, index=(cols.index(col_h1_guess) + 1) if col_h1_guess in cols else 0)

            # Opcje filtracji
            c_opt1, c_opt2 = st.columns(2)
            with c_opt1:
                ignore_query_params = st.checkbox(
                    "Ignoruj adresy URL z parametrami '?' (np. ?utm_source=...)",
                    value=True,
                    help="Odrzuca zduplikowane adresy i parametry śledzące."
                )
            with c_opt2:
                only_200 = st.checkbox(
                    "Uwzględnij wyłącznie kod HTTP 200 (pomiń błędy 404 i przekierowania 301)",
                    value=True if col_status_guess else False,
                    help="Zapewnia, że analizowane są wyłącznie działające strony docelowe."
                )

            # Filtracja rekordów
            valid_rows = []
            rejected_count = 0

            for _, row in df_raw.iterrows():
                u = str(row[col_url]).strip() if pd.notna(row[col_url]) else ""
                ct = row[col_ct_guess] if col_ct_guess and pd.notna(row[col_ct_guess]) else None
                status_val = row[col_status_guess] if (only_200 and col_status_guess and pd.notna(row[col_status_guess])) else None

                if is_valid_html_url(u, content_type=ct, http_status=status_val, ignore_query=ignore_query_params):
                    t = str(row[col_title]).strip() if col_title != "(Brak)" and pd.notna(row[col_title]) else ""
                    desc = str(row[col_desc]).strip() if col_desc != "(Brak)" and pd.notna(row[col_desc]) else ""
                    h1 = str(row[col_h1]).strip() if col_h1 != "(Brak)" and pd.notna(row[col_h1]) else ""
                    valid_rows.append((u, t, desc, h1))
                else:
                    rejected_count += 1

            st.markdown(
                f'<div class="filter-summary">✅ <b>Wynik filtracji:</b> Zakwalifikowano <b>{len(valid_rows)}</b> podstron HTML. '
                f'Odrzucono <b>{rejected_count}</b> zasobów niebędących stronami HTML (zdjęcia, pliki, skrypty, przekierowania 301 lub błędy).</div>',
                unsafe_allow_html=True
            )

            if len(valid_rows) > MAX_URL_LIMIT:
                st.warning(f"⚠️ Plik zawiera {len(valid_rows)} podstron HTML. Zgodnie z limitem bezpieczeństwa pobrano pierwsze {MAX_URL_LIMIT} URL.")
                valid_rows = valid_rows[:MAX_URL_LIMIT]

            if st.button("🚀 Uruchom analizę audytu z Clusteric", key="btn_run_clusteric", type="primary"):
                prepared_records = valid_rows

        except Exception as e:
            st.error(f"Wystąpił błąd podczas wczytywania pliku: {e}")

# --- TRYB 2: SITEMAP (MAPA WITRYNY) ---
with tab_sitemap:
    st.markdown("##### Uproszczona analiza na podstawie Mapy Witryny (Sitemap.xml)")
    st.caption("Wklej adres URL sitemapy (w tym **Sitemap Index**) LUB wgraj plik `sitemap.xml` z dysku. Narzędzie automatycznie odrzuci grafiki i pobierze wyłącznie strony HTML.")

    sitemap_source_type = st.radio(
        "Wybierz metodę wprowadzenia mapy witryny:",
        ["🌐 Podaj adres URL do sitemap.xml lub wklej listę URL", "📁 Wgraj plik sitemap.xml z dysku"],
        horizontal=True
    )

    sitemap_urls_to_analyze: List[str] = []

    if "Podaj adres URL" in sitemap_source_type:
        sitemap_input = st.text_area(
            "Adres sitemapy URL lub lista podstron:",
            height=110,
            placeholder="https://twojadomena.pl/sitemap_index.xml\nLUB: https://twojadomena.pl/sitemap.xml\nLUB wklej listę podstron..."
        )
        
        if sitemap_input.strip():
            lines = [l.strip() for l in sitemap_input.strip().splitlines() if l.strip()]
            
            # Gdy podano pojedynczy adres URL sitemapy
            if len(lines) == 1 and lines[0].startswith("http"):
                sitemap_target_url = lines[0]
                
                # Przycisk weryfikacji struktury sitemapy
                if st.button("🔎 Sprawdź strukturę sitemapy", key="btn_check_sitemap"):
                    with st.spinner("Pobieranie i badanie struktury sitemapy..."):
                        content_bytes, err = fetch_remote_content(sitemap_target_url)
                        if err == "CLOUDFLARE_403":
                            st.markdown("""
                            <div class="cf-warning-box">
                                <b>🛡️ Błąd 403 / Ochrona Cloudflare / Firewall:</b><br>
                                Serwer docelowy zablokował bezpośrednie zapytanie z chmury.<br><br>
                                <b>💡 Rozwiązanie w 10 sekund:</b>
                                <ol style="margin-bottom: 0;">
                                    <li>Otwórz link sitemapy w swojej przeglądarce i zapisz plik (skrót <code>Ctrl + S</code>).</li>
                                    <li>Wybierz opcję <b>„📁 Wgraj plik sitemap.xml z dysku”</b> powyżej i załaduj pobrany plik.</li>
                                    <li>Możesz także skorzystać z <b>Trybu 1 (Audyt z Clusteric)</b>, który działa w 100% offline bez ryzyka blokady!</li>
                                </ol>
                            </div>
                            """, unsafe_allow_html=True)
                        elif err:
                            st.error(f"Nie udało się pobrać sitemapy (kod błędu: {err}). Sprawdź czy adres jest poprawny.")
                        else:
                            parsed = inspect_and_parse_sitemap(content_bytes)
                            st.session_state["sitemap_parsed_data"] = parsed
                            st.session_state["sitemap_parent_url"] = sitemap_target_url

                # Jeśli sitemapa została sprawdzona i zapisana w sesji
                if "sitemap_parsed_data" in st.session_state and st.session_state.get("sitemap_parent_url") == sitemap_target_url:
                    parsed = st.session_state["sitemap_parsed_data"]
                    
                    if parsed["type"] == "index":
                        sub_maps = parsed["sub_sitemaps"]
                        st.success(f"📂 Wykryto **Sitemap Index** zawierający **{len(sub_maps)}** pod-map XML.")
                        
                        # Inteligentne wskazanie domyślnej pod-mapy (posty/strony/produkty)
                        def find_default_submap(sub_list):
                            for priority_keyword in ["post", "page", "artykul", "wpis", "produkt", "product"]:
                                for i, sm in enumerate(sub_list):
                                    if priority_keyword in sm.lower():
                                        return i
                            return 0

                        selected_sub = st.selectbox(
                            "Wybierz pod-sitemapę do analizy:",
                            sub_maps,
                            index=find_default_submap(sub_maps)
                        )
                        
                        with st.spinner(f"Pobieranie adresów z {selected_sub}..."):
                            sub_content, sub_err = fetch_remote_content(selected_sub)
                            if sub_err == "CLOUDFLARE_403":
                                st.error("🛡️ Ochrona Cloudflare zablokowała pobranie wybranej pod-sitemapy. Pobierz plik ręcznie i wgraj go z dysku.")
                            elif sub_err:
                                st.error(f"Błąd pobierania pod-sitemapy: {sub_err}")
                            elif sub_content:
                                sub_parsed = inspect_and_parse_sitemap(sub_content)
                                sitemap_urls_to_analyze = sub_parsed["urls"]
                                st.info(f"Odnaleziono **{len(sitemap_urls_to_analyze)}** adresów URL w wybranej pod-sitemapie.")
                    
                    elif parsed["type"] == "urlset":
                        sitemap_urls_to_analyze = parsed["urls"]
                        st.info(f"Odnaleziono **{len(sitemap_urls_to_analyze)}** adresów URL w sitemapie.")
                    else:
                        st.warning("Nie znaleziono adresów URL w podanym pliku sitemapy.")
            else:
                # Wklejono listę URL (wiele linii)
                sitemap_urls_to_analyze = [u for u in lines if u.startswith("http")]
    
    else:
        # Wgrywanie pliku z dysku
        uploaded_sitemap = st.file_uploader(
            "Wybierz plik sitemap.xml (.xml, .txt)",
            type=["xml", "txt"],
            key="uploader_sitemap_file"
        )
        if uploaded_sitemap is not None:
            file_bytes = uploaded_sitemap.read()
            parsed = inspect_and_parse_sitemap(file_bytes)
            if parsed["type"] == "index":
                st.info(f"Wykryto Sitemap Index z {len(parsed['sub_sitemaps'])} pod-mapami. Dla analizy offline zalecamy wgranie bezpośredniego pliku pod-sitemapy (np. post-sitemap.xml).")
                sitemap_urls_to_analyze = parsed["sub_sitemaps"]
            else:
                sitemap_urls_to_analyze = parsed["urls"]
                st.info(f"Odczytano **{len(sitemap_urls_to_analyze)}** adresów z pliku mapy.")

    # Wybór trybu analizy sitemapy
    scrape_mode = st.radio(
        "Tryb analizy danych z sitemapy:",
        [
            "⚡ Pełna analiza: Pobierz tagi z podstron (Title, Description, H1) [Zalecana]",
            "🚀 Szybka analiza semantyczna: Wyłącznie na podstawie struktury i ścieżek URL (bez zapytań HTTP)"
        ]
    )

    if st.button("🚀 Rozpocznij analizę Mapy Witryny", key="btn_run_sitemap", type="primary"):
        if not sitemap_urls_to_analyze:
            st.warning("Wprowadź mapę witryny i upewnij się, że adresy URL zostały załadowane.")
        else:
            # Filtracja tylko stron HTML
            filtered_urls = [u for u in sitemap_urls_to_analyze if is_valid_html_url(u, ignore_query=True)]
            excluded_count = len(sitemap_urls_to_analyze) - len(filtered_urls)

            st.markdown(
                f'<div class="filter-summary">✅ <b>Wynik filtracji sitemapy:</b> Zakwalifikowano <b>{len(filtered_urls)}</b> podstron HTML. '
                f'Odrzucono <b>{excluded_count}</b> zasobów niebędących HTML (pliki graficzne, media, parametry, inne sitemapy).</div>',
                unsafe_allow_html=True
            )

            if len(filtered_urls) > MAX_URL_LIMIT:
                st.warning(f"⚠️ Zastosowano twardy limit {MAX_URL_LIMIT} URL. Do analizy pobrano pierwsze {MAX_URL_LIMIT} podstron.")
                filtered_urls = filtered_urls[:MAX_URL_LIMIT]

            if len(filtered_urls) < 3:
                st.error("⚠️ Do przeprowadzenia analizy semantycznej wymagane są co najmniej 3 podstrony.")
            else:
                allowed, remaining = check_cooldown()
                if not allowed:
                    st.error(f"⏳ Zabezpieczenie antyspamowe: Odczekaj jeszcze {remaining} sekund przed kolejnym skanowaniem.")
                else:
                    if "Szybka analiza" in scrape_mode:
                        # Tryb błyskawiczny ze struktur URL
                        rec_list = [(u, "", "", "") for u in filtered_urls]
                        update_cooldown()
                        prepared_records = rec_list
                    else:
                        # Tryb z pobieraniem tagów HTML
                        st.write(f"Scrapowanie tagów dla **{len(filtered_urls)}** podstron HTML...")
                        p_bar = st.progress(0.0)
                        s_text = st.empty()
                        
                        scraped_data = run_scraper(filtered_urls, p_bar, s_text)
                        p_bar.empty()
                        s_text.empty()

                        rec_list = []
                        cf_blocked = 0
                        for item in scraped_data:
                            if "Blokada 403" in item.get("status", ""):
                                cf_blocked += 1
                            rec_list.append((item["url"], item["title"], item["description"], item["h1"]))
                        
                        if cf_blocked > 0:
                            st.warning(f"Uwaga: {cf_blocked} podstron zwróciło blokadę serwera (403/Cloudflare). Ich analiza została oparta na semantyce ścieżek URL.")

                        update_cooldown()
                        prepared_records = rec_list

# ==========================================
# 10. PRZETWARZANIE I PREZENTACJA WYNIKÓW
# ==========================================
if prepared_records:
    if len(prepared_records) < 3:
        st.warning("⚠️ Wymagane są co najmniej 3 podstrony do wyznaczenia centroidu.")
    else:
        st.session_state["records_tuple"] = tuple(prepared_records)

if "records_tuple" in st.session_state:
    records_tuple = st.session_state["records_tuple"]
    
    with st.spinner("🤖 Obliczanie wektorów E5 i wyznaczanie semantycznego Centroidu serwisu..."):
        results = compute_semantic_analysis(records_tuple)

    st.markdown("---")
    st.markdown("### 📊 Wyniki Analizy Topical Authority")

    # Przełącznik symulacji Content Pruning
    col_sim_title, col_sim_toggle = st.columns([3, 1])
    with col_sim_toggle:
        simulate_pruning = st.checkbox(
            "✂️ Symuluj Content Pruning",
            value=False,
            help="Wyklucza podstrony oznaczone jako 'Ryzyko Topical Drift' i przelicza na żywo nowy Centroid oraz Focus Score."
        )

    # Obliczenia bazowe
    orig_focus = results["site_focus_score"]
    orig_radius = results["site_radius"]
    orig_p90 = results["p90_radius"]
    
    is_drift = [s == "Ryzyko Topical Drift / Content Pruning" for s in results["statuses"]]
    num_drift = sum(is_drift)
    total_pages = len(records_tuple)

    # Obsługa symulacji Pruningu
    if simulate_pruning and num_drift > 0 and (total_pages - num_drift) >= 2:
        keep_indices = [i for i, d in enumerate(is_drift) if not d]
        pruned_embeddings = results["embeddings"][keep_indices]
        
        p_centroid = np.mean(pruned_embeddings, axis=0)
        p_norm = np.linalg.norm(p_centroid)
        p_centroid = p_centroid / (p_norm if p_norm > 0 else 1.0)
        
        p_sim = np.dot(pruned_embeddings, p_centroid)
        p_sim = np.clip(p_sim, -1.0, 1.0)
        p_radii = 1.0 - p_sim
        
        active_radius = float(np.mean(p_radii))
        active_raw_sim = 1.0 - active_radius
        active_focus = float(np.clip((active_raw_sim - 0.65) / (0.92 - 0.65) * 100.0, 0.0, 100.0))
        
        delta_focus = active_focus - orig_focus
        delta_radius = active_radius - orig_radius
    else:
        active_focus = orig_focus
        active_radius = orig_radius
        delta_focus = None
        delta_radius = None

    # KARTY METRYK (3 karty na górze)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="🎯 Site Focus Score",
            value=f"{active_focus:.1f}%",
            delta=f"+{delta_focus:.1f} pp." if delta_focus is not None else None,
            help="Spójność tematyczna witryny mierzona dopasowaniem do Centroidu. Skalowanie [0.65, 0.92] -> 0-100%."
        )
    with m2:
        st.metric(
            label="📐 Średni Site Radius",
            value=f"{active_radius:.4f}",
            delta=f"{delta_radius:.4f}" if delta_radius is not None else None,
            delta_color="inverse",
            help="Średnia odległość podstron od centroidu (1 - cosine similarity). Niższa wartość oznacza ściślejszy profil witryny."
        )
    with m3:
        st.metric(
            label="🚩 Podstrony rozmywające (Drift)",
            value=f"{num_drift} / {total_pages}",
            delta=f"-{num_drift} wykluczonych" if simulate_pruning else None,
            delta_color="inverse",
            help="Liczba podstron o najwyższym odchyleniu od tematu głównego (kandydaci do usunięcia, kanonizacji lub przebudowy)."
        )

    if simulate_pruning and delta_focus is not None:
        st.success(
            f"📈 **Efekt symulacji Content Pruning:** Po wycięciu **{num_drift}** podstron o najwyższym Topical Drift, "
            f"spójność domeny **Site Focus Score** wzrosła o **+{delta_focus:.1f} pp.** (z {orig_focus:.1f}% do {active_focus:.1f}%), "
            f"a średni promień witryny spadł z **{orig_radius:.4f}** do **{active_radius:.4f}**."
        )

    st.markdown("---")

    # ==========================================
    # 11. WIZUALIZACJA PCA 2D (PLOTLY)
    # ==========================================
    st.markdown("#### 🗺️ Semantyczna Mapa Witryny (Redukcja PCA 384D ➔ 2D)")
    st.caption("Interaktywny wykres punktowy przedstawiający rozkład podstron w przestrzeni wektorowej względem wyznaczonego Centroidu.")

    plot_df = pd.DataFrame({
        "url": results["urls"],
        "title": results["titles"],
        "description": results["descriptions"],
        "h1": results["h1s"],
        "clean_path": results["clean_paths"],
        "radius": results["page_radius"],
        "similarity": results["cosine_sim"],
        "status": results["statuses"],
        "pca_x": results["pca_x"],
        "pca_y": results["pca_y"],
        "is_pruned": is_drift if simulate_pruning else [False] * total_pages
    })

    fig = go.Figure()

    status_order = [
        ("Rdzeń tematyczny (Core)", "#10B981", "circle"),
        ("Temat poboczny (Supporting)", "#F59E0B", "circle"),
        ("Ryzyko Topical Drift / Content Pruning", "#EF4444", "diamond")
    ]

    for status_name, color, symbol in status_order:
        subset = plot_df[plot_df["status"] == status_name]
        if not subset.empty:
            opacity = 0.22 if (simulate_pruning and status_name == "Ryzyko Topical Drift / Content Pruning") else 0.85
            hover_text = [
                f"<b>{row['url']}</b><br>"
                f"🏷️ <b>Tytuł:</b> {row['title'] if row['title'] else '(brak)'}<br>"
                f"📝 <b>Opis:</b> {(row['description'][:85] + '...') if row['description'] else '(brak)'}<br>"
                f"📌 <b>H1:</b> {row['h1'] if row['h1'] else '(brak)'}<br>"
                f"📏 <b>Page Radius:</b> {row['radius']:.4f}<br>"
                f"🎯 <b>Podobieństwo:</b> {row['similarity']:.4f}<br>"
                f"🚩 <b>Status:</b> {row['status']}"
                for _, row in subset.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=subset["pca_x"],
                y=subset["pca_y"],
                mode="markers",
                name=f"{status_name} ({len(subset)})" + (" [WYKLUCZONE]" if (simulate_pruning and status_name == "Ryzyko Topical Drift / Content Pruning") else ""),
                marker=dict(
                    size=10,
                    color=color,
                    symbol=symbol,
                    opacity=opacity,
                    line=dict(width=1, color="#1E293B")
                ),
                text=hover_text,
                hoverinfo="text"
            ))

    # Dodanie Centroidu (złota gwiazda)
    fig.add_trace(go.Scatter(
        x=[results["centroid_pca_x"]],
        y=[results["centroid_pca_y"]],
        mode="markers+text",
        name="⭐ Centroid Witryny (Topical Center)",
        text=["⭐ CENTROID"],
        textposition="top center",
        textfont=dict(size=12, color="#D97706", family="sans-serif"),
        marker=dict(
            symbol="star",
            size=22,
            color="#FFD700",
            line=dict(width=2, color="#92400E")
        ),
        hoverinfo="text",
        hovertext="<b>⭐ CENTROID TEMATYCZNY SERWISU</b><br>Środek ciężkości wektorów wszystkich podstron."
    ))

    fig.update_layout(
        template="plotly_white",
        height=580,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        xaxis=dict(title="Główna składowa PCA 1", showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(title="Główna składowa PCA 2", showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 12. TABELA WYNIKOWA I EKSPORT
    # ==========================================
    st.markdown("#### 📋 Szczegółowy Audyt Podstron")
    st.caption("Tabela posortowana malejąco według wskaźnika Page Radius (od podstron najbardziej oddalonych od tematu witryny).")

    table_df = pd.DataFrame({
        "Status": results["statuses"],
        "Page Radius": [round(r, 4) for r in results["page_radius"]],
        "Podobieństwo (Cos Sim)": [round(s, 4) for s in results["cosine_sim"]],
        "URL": results["urls"],
        "Title": results["titles"],
        "Description": results["descriptions"],
        "H1": results["h1s"]
    })

    table_df = table_df.sort_values(by="Page Radius", ascending=False).reset_index(drop=True)

    st.dataframe(
        table_df,
        use_container_width=True,
        column_config={
            "Status": st.column_config.TextColumn("Status Semantyczny"),
            "Page Radius": st.column_config.NumberColumn("Page Radius", format="%.4f"),
            "Podobieństwo (Cos Sim)": st.column_config.NumberColumn("Podobieństwo", format="%.4f"),
            "URL": st.column_config.LinkColumn("Adres URL"),
            "Title": st.column_config.TextColumn("Tytuł <title>"),
            "Description": st.column_config.TextColumn("Meta Description"),
            "H1": st.column_config.TextColumn("Nagłówek <h1>")
        },
        height=420
    )

    csv_bytes = table_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 Pobierz pełny raport CSV",
        data=csv_bytes,
        file_name="site_focus_score_report.csv",
        mime="text/csv",
        type="secondary"
    )

# ==========================================
# 13. SEKCJA CALL TO ACTION (CTA)
# ==========================================
st.markdown("""
<div class="cta-container">
    <div class="cta-title">Chcesz przeanalizować cały serwis (powyżej 150 URL) i wdrożyć architekturę semantyczną?</div>
    <div class="cta-desc">
        Większe serwisy e-commerce oraz rozbudowane portale wymagają zaawansowanego klastrowania wektorowego, 
        optymalizacji linkowania wewnętrznego oraz bezpiecznego audytu Content Pruning bez utraty ruchu organicznego.
    </div>
    <a href="https://janwojcicki.pl/#kontakt" target="_blank" class="cta-btn">
        📩 Skontaktuj się ze mną
    </a>
</div>
""", unsafe_allow_html=True)
