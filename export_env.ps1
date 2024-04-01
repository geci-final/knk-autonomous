conda env export --no-builds | Select-String -NotMatch "^prefix: " | Out-File -Encoding utf8 environment.yml
echo "Environment exported to environment.yml"