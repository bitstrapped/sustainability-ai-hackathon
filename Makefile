build:
	source envs/.env
	scripts/build.sh

push:
	source envs/.env
	scripts/push.sh

deploy:
	source envs/.env
	scripts/deploy.sh