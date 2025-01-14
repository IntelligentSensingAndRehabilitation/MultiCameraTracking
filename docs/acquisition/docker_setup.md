# Docker Setup

## Install Docker according to the [Docker website](https://docs.docker.com/engine/install/ubuntu/)

Copy and paste each of these blocks into the terminal and run them

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL <https://download.docker.com/linux/ubuntu/gpg> | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

```
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```
sudo docker run hello-world
```

## Confirm the docker group exists (or run the following)

```
sudo groupadd docker
```

## Make sure the current user is added to the docker group

```
sudo usermod -aG docker $USER
```

## Apply changes

Can log out/log back in or run

```
newgrp docker
```

## Confirm docker commands can be run without sudo

```
docker run hello-world
```