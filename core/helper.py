def spacy_download():
    import sys, subprocess
    logger.debug("Installing 'spacy' module")
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'])