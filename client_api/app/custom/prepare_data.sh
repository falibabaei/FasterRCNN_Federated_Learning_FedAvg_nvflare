

python3 -c "from roboflow import Roboflow;
rf = Roboflow(api_key="MGBDglstJuI7h1wLl8Hg");
project = rf.workspace("bronkscottema").project("football-player-detection");
dataset = project.version(3).download("voc")"
