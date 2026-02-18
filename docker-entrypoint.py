"""Docker entrypoint that suppresses noisy Essentia warnings before running the tagger."""
import essentia.log
essentia.log.warningActive = False
essentia.log.infoActive = False

import runpy
runpy.run_path("tag_music.py", run_name="__main__")
