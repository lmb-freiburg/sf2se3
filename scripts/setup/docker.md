# Docker Install

##### 1. remove old docker installations
`sudo snap remove docker --purge`  
`sudo apt-get remove docker`  
`sudo apt-get remove docker-engine`  
`sudo apt-get remove docker.io`  
`sudo apt-get remove containerd`  
`sudo apt-get remove runc`  

##### 2. install dependencies
`sudo apt-get update`  
`sudo apt-get install ca-certificates curl gnupg lsb-release`  

##### 3. add gpg key
`curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`

##### 4. add repository
`echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`  

##### 5. install docker
`sudo apt-get update`  
`sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin`  

##### 6. install nvidia docker (for gpu use)
`distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`  
`sudo apt-get update`  
`sudo apt-get install -y nvidia-docker2`  
`sudo systemctl restart docker`  

##### 7. in case of retrieving gpg error time out while building image
put `keyserver keys.openpgp.org` into `~/.gnupg/gpg.conf.`  



# Docker Commands

##### 1. list containers
`docker ps -a`  
##### 2. delete all stopped containers
`docker container prune`  
##### 3. delete all dangling images (images that are not tagged or referenced by any container)
`docker image prune`  
##### 4. delete all images that are not used by existing containers
`docker image prune -a`  
