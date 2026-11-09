
build-dev:
	docker build -t drcm .

run-dev:
	aws configure export-credentials --profile developers --format env-no-export > .env_list
	docker run --rm -it --env-file .env_list --entrypoint bash drcm:latest

