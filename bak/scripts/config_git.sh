git config --local user.email zhoushenglong@pjlab.org.cn
git config --local user.name zhoushenglong
git config --local core.sshCommand "ssh -i ~/.ssh/id_rsa"
git config --local url.ssh://git@github.com/.insteadOf https://github.com/
#git config --local core.excludesFile /home/cse/zhousl/.gitignore_zhousl
git submodule foreach --recursive \
	    'git config --local core.sshCommand "ssh -i ~/.ssh/id_rsa"'
git submodule foreach --recursive \
	    'git config --local url.ssh://git@github.com/.insteadOf https://github.com/'

git config --global user.email zhoushenglong@pjlab.org.cn
git config --global user.name zhoushenglong
git config --global core.sshCommand "ssh -i ~/.ssh/id_rsa"
git config --global url.ssh://git@github.com/.insteadOf https://github.com/
#git config --global core.excludesFile /home/cse/zhousl/.gitignore_zhousl
git submodule foreach --recursive \
	    'git config --global core.sshCommand "ssh -i ~/.ssh/id_rsa"'
git submodule foreach --recursive \
	    'git config --global url.ssh://git@github.com/.insteadOf https://github.com/'

