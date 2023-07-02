conda env remove --name spotcheck
yes | conda create --name spotcheck python=3.7
source activate spotcheck
pip install git+https://github.com/openai/CLIP.git
pip install "domino[clip,text] @ git+https://github.com/HazyResearch/domino@main"
pip install torchvision
conda list --name spotcheck > env.txt

