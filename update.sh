if [ -f "model.ckpt" ]; then
    rm "model.ckpt"
fi

git stash
git stash drop
git pull
pip install -r requirements.txt