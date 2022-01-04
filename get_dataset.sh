# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VNuA9NLE7Qc_Xs3hLqnbY0sXhzBrPiPK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VNuA9NLE7Qc_Xs3hLqnbY0sXhzBrPiPK" -O food_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./food_data.zip

# Remove the downloaded zip file
rm ./food_data.zip
