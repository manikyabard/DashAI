%cd app/
npm install
npm run electron-pack
%cd ..
conda create -n server python=3.6
%conda activate server
pip install -r requirements.txt