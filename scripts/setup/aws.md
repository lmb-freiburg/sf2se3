
### Install AWS CLI
`curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"`  
`unzip awscliv2.zip`  
`sudo ./aws/install`  

### Setup AWS Image

aws configure
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 603496633769.dkr.ecr.eu-west-2.amazonaws.com