# Windows Modifications

The annotation system expects a Linux OS by default. If you 
are running on Windows or WSL the following modifications are
needed.

## IP Address
- Update datajoint host name to the IP address of where your database is running

## Docker Compose
- Remove `network_mode: host` from the annotation service in the docker-compose
- Add the following to the annotation service to expose the necessary ports:
    ```
    ports:
      - "3005:3005"
      - "8005:8005"
    ```

## Additional Modifications if your database is running from the datajoint_docker from PosePipeline
- Create a docker bridge network: `docker network create <network_name>`
- Add the following to the docker-compose for the datajoint_docker
    ```
    networks:
      <network_name>:
        driver: bridge
    ```
- Add the following to the docker-compose for the annotation service
    ```
    networks: 
      <network_name>:
        external: true
    ```


## Run Command
- To start annotation, run `docker compose up annotation`
