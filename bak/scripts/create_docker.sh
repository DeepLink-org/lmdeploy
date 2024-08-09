NET_ARGS="--net=host"
CONTAINER_NAME=zhousl_card_4_7
WORK_DIR=/home/costest/zhousl
HOME_DIR=${WORK_DIR}
MODEL_DIR=/data/models

docker run -itd ${NET_ARGS} \
	--ipc=host  \
	--cap-add SYS_PTRACE  \
	--device=/dev/mxcd \
	--device=/dev/dri/card0 \
	--device=/dev/dri/card1 \
	--device=/dev/dri/card2 \
	--device=/dev/dri/card3 \
	--device=/dev/dri/card4 \
	--device=/dev/dri/card5 \
	--device=/dev/dri/card6 \
	--device=/dev/dri/card7 \
	--device=/dev/dri/renderD128 \
	--device=/dev/dri/renderD129 \
	--device=/dev/dri/renderD130 \
	--device=/dev/dri/renderD131 \
	--device=/dev/dri/renderD132 \
	--device=/dev/dri/renderD133 \
	--device=/dev/dri/renderD134 \
	--device=/dev/dri/renderD135 \
	--device=/dev/infiniband \
	--shm-size 100G --ulimit memlock=-1 \
	-h zhousl_dev_docker \
	--name ${CONTAINER_NAME} \
	-v ${MODEL_DIR}:${MODEL_DIR} \
	-v ${HOME_DIR}:${HOME_DIR} \
	-w ${WORK_DIR} \
	--entrypoint /bin/bash \
	9cf590c90bf3
