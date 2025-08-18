#!/bin/bash
# Ensure the devnet Docker network exists

# Check if devnet network exists
if ! docker network ls | grep -q "devnet"; then
    echo "Creating devnet network..."
    docker network create devnet
else
    echo "devnet network already exists"
fi