wget 'https://www.dropbox.com/scl/fo/8a8gw5yk8j2hh1lj2igsa/h?rlkey=dlrctcwld2jilbbcws869sjlu&dl=1' -O best.zip && unzip best.zip -d best && rm best.zip
wget 'https://www.dropbox.com/scl/fo/08bm7qf7qz5qhmm7ycj8u/h?rlkey=tuaw0n6f5k35qfsumhd7hldte&dl=1' -O cache.zip && unzip cache.zip -d cache && rm cache.zip
wget 'https://www.dropbox.com/scl/fo/607l5k3vzq3i15mna3sod/h?rlkey=z2voc6n5vrzkxik4om5zzywht&dl=1' -O sub.zip && unzip sub.zip -d sub && rm sub.zip    
wget 'https://www.dropbox.com/scl/fi/k5i77fzbdwjdsd4v0zre4/train.sh?rlkey=fllzotpd3sgo69u8524ckg4t3&dl=1' -O train.sh
wget 'https://www.dropbox.com/scl/fi/jyujs4u46ix2pkpo1ykdy/test.sh?rlkey=p9ihnp9lsxqxuvw0o8fl7lg36&dl=1' -O test.sh
wget 'https://www.dropbox.com/scl/fi/sohgpmunz1rts41qlzxt4/eval.sh?rlkey=rcylz1zmmpjhnf367ynqkmm7v&dl=0' -O eval.sh
git clone https://huggingface.co/google/mt5-small