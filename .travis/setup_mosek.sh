#!/bin/sh -e

. .travis/utility.sh

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