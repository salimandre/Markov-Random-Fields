#!/bin/bash

git pull https://github.com/salimandre/Markov-Random-Fields.git
git add .
git commit -m "$1"
git push https://github.com/salimandre/Markov-Random-Fields.git master
