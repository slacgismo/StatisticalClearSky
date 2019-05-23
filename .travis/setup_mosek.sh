#!/bin/sh -e
# vim: set tabstop=2 shiftwidth=2 noexpandtab:

### 
# Setup some styles
###
sr=$(echo -e "\033[;31m") # red
sg=$(echo -e "\033[;32m") # green
sm=$(echo -e "\033[;35m") # magenta
sw=$(echo -e "\033[;37m")
sb=$(echo -e "\033[1m")
sn=$(echo -e "\033[0;0m") # reset

### 
# Start a logically related section with the above stylings - a simple utility function
###
function start_section {
    if [ "x$RELLIB_SECTION" = "x" ]; then
        RELLIB_SECTION=0
    fi

    message=$1

    if [ $RELLIB_SECTION = 0 ]; then
        describe_action "$@"
        RELLIB_SECTION=1
    else
        if [ $RELLIB_SECTION -lt 5 ]; then
            colors=($sm $sw $sb)
            echo "${colors[$((RELLIB_SECTION - 1))]}$RELLIB_SECTION${sn}> $message"
        else
            echo "$RELLIB_SECTION| $message"
        fi
        RELLIB_SECTION=$((RELLIB_SECTION + 1))
    fi
}

### 
# End a logically related section 
###
function end_section {
    if [ "$RELLIB_SECTION" = "0" ]; then
       describe_action "ERROR: unexpected end_section"
       exit -255
    else
        RELLIB_SECTION=$((RELLIB_SECTION - 1))
    fi
}

###
# Convenience to verify our custom environment variables are properly loaded
###
function check_for_env {
    for envvar in $@; do
        if [ -z "${!envvar}" ]; then
            echo "${sr}${sb}ERROR${sn}: Expected environment variable $envvar to be present and not empty" 1>&2
            exit 1
        fi
        echo "${sb}$envvar${sn} defined"
    done
}

start_section "Create the mosek license folder"
    cd $HOME
    mkdir mosek
end_section

start_section "Verify the AWS creds are loaded in the environment"
    check_for_env AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
end_section

start_section "Pull the mosek license from s3"
    aws s3 cp s3://slac.gismo.ci.artifacts/mosek.license/mosek.lic $HOME/mosek/mosek.lic
end_section