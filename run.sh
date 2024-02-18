#!/bin/bash

#SBATCH -t 1:00 
#SBATCH --mem=256M 
#SBATCH -J train_bert 
#SBATCH -p all
#SBATCH -c 1 
#SBATCH -N 1 
#SBATCH -o %x-%j.out 

echo "Hello"
echo "test out" > test.txt

python3 bert.py
