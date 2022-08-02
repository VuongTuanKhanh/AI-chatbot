def spacy_download():
    import sys, subprocess
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'])

def scikitplot_install():
    import sys, subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-plot'])