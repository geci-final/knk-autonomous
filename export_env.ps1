conda env export --no-builds | grep -v "^prefix: " > environment.yml
echo "Environment exported to environment.yml"