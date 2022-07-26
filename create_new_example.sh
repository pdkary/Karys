#!/bin/bash

EXAMPLE_NAME=$1
EXAMPLE_SETUP_FILE=$(sed 's/\(.*\)/"\1",/g' examples/example_setup_script.py)

read -r -d '' EXAMPLE_NB_PREFIX <<-EOM
{
 "cells": [
     {"cell_type":"markdown","metadata":{},"source":["### ${EXAMPLE_NAME^}"]},
     {
         "cell_type": "code",
         "execution_count": null,
         "metadata": {},
         "outputs": [],
         "source": [
EOM

read -r -d '' EXAMPLE_NB_SUFFIX <<-EOM
        ]
    }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
EOM

mkdir examples/$EXAMPLE_NAME
mkdir examples/$EXAMPLE_NAME/test_input
mkdir examples/$EXAMPLE_NAME/test_output
echo $EXAMPLE_NB_PREFIX ${EXAMPLE_SETUP_FILE::-1} $EXAMPLE_NB_SUFFIX >examples/$EXAMPLE_NAME/$EXAMPLE_NAME.ipynb
