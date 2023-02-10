#!/bin/bash
#
# Helper script for attaching to a server via ssh, rsync, etc.
#
# Args:
#  <user@server-name-or-ip>
#  --ssh
#      * ssh to the server
#  --launch command
#      * run command on the server (through nohup)
#  -d, --deploy [server-root="~/dev"]
#      * rsync the current working directory to the server with --ignore-existing flag,
#      * optionally adjusting the destination server root directory
#  --deploy-force [server-root="~/dev"]
#      * rsync the current working directory to the server without --ignore-existing flag,
#      * optionally adjusting the destination server root directory
#  -t, --tunnel local_port server_port
#      * establish a tunnel to the server that forwards the port
#        * where local_port specifies the port on the local machine
#        * and server_port specifies on the server to forward to the local port
#      * repeat the argument to forward multiple ports
#  -g, --grab server_filespec [local_dir=.]
#      * rsync the server filespec to the local dir (. by default) w/--ignore-existing
#      * NOTE: currently a server filespec with spaces cannot be "grabbed"
#  --grab-force server_filespec [local_dir=.]
#      * rsync the server filespec to the local dir (. by default) w/out --ignore-existing
#      * NOTE: currently a server filespec with spaces cannot be "grabbed"
#  -r, --server-root
#      * specify the server root directory (default "~/dev")
#  -e, --exclude exclude_pattern
#      * add the given pattern to excludes
#  -c, --clear-excludes
#      * clear all preceding excludes (e.g., default excludes)
#  --open-tunnel
#      * use the DEV_* env vars to open a (notebook) port forwarding tunnel
#  --dryrun
#      * echo the commands that would be executed without running them
#
# Examples:
#
# Copy the ~/abc/foo directory to "user@server":dev/foo with default project excludes:
#
# $ cd ~/dev/foo
# $ attach -d
#
# Establish a tunnel to "user@server" exposing server ports 8888 and 8080 as local 28888 and 28080:
#
# $ attach user@server -t 28888 8888 -t 28080 8080
#
# Copy (grab) the test results from the server to the localhost
#
# $ attach user@server -g _testing_output
#

#set -e

if test -e .env; then
    . .env;
fi

user="$DEV_USER"
server="$DEV_SERVER"

# NOTE: An initial argument of the form user@server overrides env vars
if [[ "$1" =~ '@' ]]; then
    IFS="@" read -r -a user_at_server <<< "$1";
    shift;

    if [[ ${#user_at_server[@]} == 2 ]]; then
        user=${user_at_server[0]};
        server=${user_at_server[1]};
    fi
fi

DEFAULT_EXCLUDES='*~ *# .git __pycache__ .pytest_cache *.egg-info .ipynb_checkpoints .aws';
excludes="";
for e in $DEFAULT_EXCLUDES; do
    excludes="$excludes --exclude $e";
done;
    
tunnel="";
open_tunnel="";
if test -n "$DEV_NOTEBOOK_PORT" && test -n "$DEV_SERVER"; then
    nb_tunnel="-L $DEV_NOTEBOOK_PORT:127.0.0.1:$DEV_NOTEBOOK_PORT";
else
    nb_tunnel="";
fi;

do_ssh="";
launch_cmd="";
server_root="${DEV_SERVER_ROOT:-dev}"
deploy="";
grab="";
dryrun="";
rsync_flags="";

# parse args
while [[ $# > 0 ]]
do
    key="$1";

    case "$key" in
        --ssh)
            do_ssh="y";
            shift;
            ;;

        --launch)
            launch_cmd="$2";
            shift 2;
            ;;

        --deploy|-d)
           deploy="y";
           rsync_flags="--ignore-existing";
           shift;
           if [[ $# > 0 ]] && ! [[ "$1" =~ "^-" ]]; then
               # optional server_root is present
               server_root="$1"
               shift;
           fi;
           ;;

        --deploy-force)
           deploy="y";
           rsync_flags="--delete";
           shift;
           if [[ $# > 0 ]] && ! [[ "$1" =~ "^-" ]]; then
               # optional server_root is present
               server_root="$1"
               shift;
           fi;
           ;;

        --tunnel|-t)
           tunnel="$tunnel -L $2:127.0.0.1:$3"
           shift 3;
           ;;

        --open-tunnel)
           tunnel="$tunnel $nb_tunnel"
           shift;
           ;;

        --grab|-g)
           grab="$2";
           rsync_flags="--ignore-existing";
           shift 2;
           ;;

        --grab-force)
           grab="$2";
           rsync_flags="";
           shift 2;
           ;;

        --server-root|--root|-r)
            server_root="$2"
            shift;
            ;;

        --exclude|-e)
           excludes="$excludes --exclude $2";
           shift 2;
           ;;
               

        --clear-excludes|-c)
            excludes="";
            shift;
            ;;

        --dry*)
            dryrun="y";
            shift;
            ;;

        *)
            shift;

   esac;
done;

echo "user=$user"
echo "server=$server"
echo "deploy=$deploy"
echo "server_root=$server_root"
echo "tunnel=$tunnel"
echo "grab=$grab"
echo "excludes=$excludes"
echo "rsync_flags=$rsync_flags"
echo "dryrun=$dryrun"


srcdir="`basename $PWD`"
destdir="$server:${server_root%/}/"

#TODO: Figure out how to incorporate this (probably just use .ssh config
ssh_cmd="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o User=$user"

# Rsync current folder to server
if test -n "$deploy"; then
    #rsync_command="rsync -e \"$ssh_cmd\" $excludes -avz $srcdir $destdir"
    rsync_command="rsync $excludes -avz ${rsync_flags} $srcdir ${user}@${destdir}"
    echo "$rsync_command";
    if test -z "$dryrun"; then
        pushd ..;
        $rsync_command;
        popd;
    fi;
fi;


# Open a port forwarding tunnel
if test -n "$tunnel"; then
    tunnel_command="ssh ${user}@${server} $tunnel -N";
    echo "$tunnel_command";
    if test -z "$dryrun"; then
        $tunnel_command;
    fi;
fi;


# Grab (rsync) a filespec from the server to localhost
if test -n "$grab"; then
    #TODO: fix for if there is a space in the file to grab
    #grab=`echo "$grab" | sed 's! !\\\ !g'`
    rsync_command="rsync $excludes -avz ${rsync_flags} ${user}@${destdir}${srcdir}/${grab} ./${grab}"
    echo "$rsync_command";
    if test -z "$dryrun"; then
        $rsync_command;
    fi;
fi;


# ssh to the server
if test -n "$do_ssh"; then
    ssh_command="ssh ${user}@${server} -t cd ${server_root}/${srcdir}; bash --login"
    echo "$ssh_command";
    if test -z "$dryrun"; then
        $ssh_command;
    fi;
fi;


# Launch a process on the server
if test -n "$launch_cmd"; then
    ssh_command="ssh ${user}@${server} -t cd ${server_root}/${srcdir}; $launch_cmd"
    echo "$ssh_command";
    if test -z "$dryrun"; then
        $ssh_command;
    fi;
fi;
