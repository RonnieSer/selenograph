"""Browse installed maps, annotations, and verification metadata.

Verification is dataset or user metadata, not independent certification.
Previews render in memory; verification changes are staged as drafts.
"""
import os

import streamlit as st
from PIL import Image

from app import core, pictures
from app.core import CMAP, CODES

COLS = 4
PAGE_SIZE = 24


def chip(code, label):
    """Return an HTML colour swatch and label for a geologic unit."""
    rgb = "#%02x%02x%02x" % CMAP[code][:3]
    return (f'<span style="display:inline-block;width:0.72em;height:0.72em;'
            f'background:{rgb};border-radius:2px;margin-right:0.4em;'
            f'vertical-align:-0.03em"></span>{label}')


def load_all():
    """Collect installed map metadata and annotation availability."""
    painted, _ = core.painted_maps()
    maps = []
    for dem in core.dem_files():
        name = core.dem_name(dem)
        bundled_labels = core.annotation_path(name, "labels")
        saved_labels = os.path.isfile(core.output_file(name, "labels.tif", existing=True))
        labeled = bool(name in painted or saved_labels or
                       (bundled_labels and os.path.isfile(bundled_labels)))
        record = core.map_record(name) or {"working_dem": dem}
        maps.append(dict(core.meta_get(name), name=name, _dem=dem,
                         _record=record, _section=str(core.section(record)),
                         _saved_labels=saved_labels, _labeled=labeled,
                         _saved_painted=core.has_saved_labels(name)))
    return maps


def preview_stamp(name, dem):
    """Return source file revisions used to invalidate cached previews."""
    # Adding or removing a draft changes which painting supplies the preview.
    paths = [dem, core.annotation_path(name, "painting"), core.output_file(name, "thumb.png", existing=True)]
    paths += core.manifest_files()
    paths += [core.painting_files(name, final, existing=True)[0] for final in (False, True)]
    stamps = []
    for path in paths:
        if path and os.path.isfile(path):
            stat = os.stat(path)
            stamps.append((str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
    return tuple(stamps)


@st.cache_data(show_spinner=False, max_entries=96)
def thumbnail(name, dem, revision, saved_only=False):
    """Render a terrain and annotation preview without loading the model."""
    elevation, transform, _, pixel_size = core.read(dem, max_side=240)
    filled, missing = core.fill_nan(elevation)
    thumbnail_path = core.output_file(name, "thumb.png", existing=True)
    cells, source = core.painting_source(name, final=True if saved_only else None)
    # A saved screenshot can contain an old prediction or an older painting.
    # Render current cells whenever available, including an explicitly empty draft.
    if source is None and os.path.isfile(thumbnail_path):
        with Image.open(thumbnail_path) as saved:
            image = saved.convert("RGB")
    else:
        base = core.hillshade(filled, pixel_size)
        image = pictures.render(base, [(cells, 0.6)])
    bounds = pictures.tile_bounds(name, elevation.shape, transform)
    return pictures.png(pictures.gallery_preview(image, bounds=bounds, missing=missing))


st.subheader("Maps")

maps = load_all()
if not maps:
    st.info("No installed maps yet.")
    st.stop()

sections = ["All", *sorted({item["_section"] for item in maps})]
if st.session_state.get("gallery_section") not in sections:
    st.session_state.gallery_section = "All"
with st.container(horizontal=True, vertical_alignment="bottom"):
    section = st.selectbox("Section", sections, key="gallery_section", width=300,
                           format_func=lambda value: "All sections" if value == "All" else value)
    status = st.segmented_control("Maps", ["All maps", "Verified only", "Saved, unverified"],
                                  default="All maps", key="gallery_status", required=True,
                                  help="Saved, unverified excludes draft-only paintings.")
if section != "All":
    maps = [item for item in maps if item["_section"] == section]
if status == "Verified only":
    maps = [item for item in maps if item.get("verified") is True]
    if not maps:
        st.info("No verified maps in this selection. Tick **Verified** here or in Paint.")
        st.stop()
elif status == "Saved, unverified":
    maps = [item for item in maps if item["_saved_painted"] and item.get("verified") is not True]
    if not maps:
        st.info("No saved, unverified paintings in this selection.")
        st.stop()

filters = (section, status)
page_count = (len(maps) + PAGE_SIZE - 1) // PAGE_SIZE
if st.session_state.get("gallery_filters") != filters:
    st.session_state.gallery_page = 1
st.session_state.gallery_filters = filters
st.session_state.gallery_page = min(max(st.session_state.get("gallery_page", 1), 1), page_count)
page = 1
if page_count > 1:
    page = st.selectbox("Page", range(1, page_count + 1), key="gallery_page", width=180,
                        format_func=lambda value: f"{value} of {page_count}")
start = (page - 1) * PAGE_SIZE
visible = maps[start:start + PAGE_SIZE]
verified_count = sum(item.get("verified") is True for item in maps)
st.caption(f"Showing {start + 1}–{start + len(visible)} of {len(maps)} maps · {verified_count} verified")

for row in range(0, len(visible), COLS):
    for column, item in zip(st.columns(COLS), visible[row:row + COLS]):
        name = item["name"]
        record = item["_record"]
        with column, st.container(border=True, gap="xsmall"):
            try:
                st.image(thumbnail(name, item["_dem"], preview_stamp(name, item["_dem"]),
                                   saved_only=status == "Saved, unverified"),
                         width="stretch")
            except (OSError, ValueError) as error:
                st.info(f"Painting preview is unavailable: {error}")
            st.markdown(f"**{name}**")
            if item["_saved_painted"] or item["_saved_labels"] or record.get("annotations"):
                state = "Saved labels"
            elif item["_labeled"]:
                state = "Working painting"
            else:
                state = "No labels yet"
            st.caption(f"{item['_section']} · {state}")

            read_only = bool(record.get("read_only"))
            editable = not read_only
            verified = st.checkbox("Verified",
                                   value=bool(item.get("verified")), key=f"v_{name}",
                                   disabled=not editable,
                                   help=("This map is read only." if read_only else
                                         "Use this map for training. Save in Paint to commit."))
            if editable and verified != bool(item.get("verified")):
                core.meta_set(name, final=False, verified=verified)
                st.rerun()
            if item.get("saved_at") or item.get("painted_area_km2") or "painted_frac" in item:
                with st.expander("Details"):
                    if item.get("saved_at"):
                        st.caption(f"Saved {item['saved_at']}")
                    areas = item.get("painted_area_km2", {})
                    for c, unit in CODES.items():
                        if unit in areas:
                            st.markdown(chip(c, f"{unit.replace('_', ' ')} — {areas[unit]:g} km²"),
                                        unsafe_allow_html=True)
                    if "painted_frac" in item:
                        caption = f"{item['painted_frac'] * 100:.1f} % labeled at last save"
                        if item.get("model"):
                            caption += f" · {item['model']}"
                        st.caption(caption)
