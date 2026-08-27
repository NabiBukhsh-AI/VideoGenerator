# Caption fonts

Drop a `.ttf` or `.ttc` here and `videogen` will prefer it for burned-in captions.

Font resolution order (see `videogen/render/fonts.py`):

1. `--font` on the CLI, or `VIDEOGEN_FONT_PATH`
2. the first font in this directory
3. a platform font (Segoe UI Bold, Arial Bold, Helvetica, DejaVu Sans Bold, …)
4. matplotlib's bundled DejaVu, if matplotlib is installed
5. PIL's built-in bitmap font (a legible last resort; ignores size)

No font is committed here because most are not redistributable under this project's
licence. A heavy weight reads best as a caption — DejaVu Sans Bold and Inter Bold are
both good, freely licensed choices.
