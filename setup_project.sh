#!/bin/bash

# Function to display a cool welcome logo
show_logo() {
    echo "================================================================================"
    echo -e "\033[1;32m🚀 $1 🚀\033[0m"  # Green color for the project name
    echo "================================================================================"
    echo " ____  _ _          _____ _             _   _____           _           _      "
    echo "|  _ \(_) |        / ____(_)           | | |  __ \         (_)         | |     "
    echo "| |_) |_| |_ ___  | (___  _ _______  __| | | |__) | __ ___  _  ___  ___| |_    "
    echo "|  _ <| | __/ _ \  \___ \| |_  / _ \/ _\` | |  ___/ '__/ _ \| |/ _ \/ __| __|   "
    echo "| |_) | | ||  __/  ____) | |/ /  __/ (_| | | |   | | | (_) | |  __/ (__| |_    "
    echo "|____/|_|\__\___| |_____/|_/___\___|\__,_| |_|   |_|  \___/| |\___|\___|\__|   "
    echo "                                                         _/ |                  "
    echo "                                                        |__/                   "
    echo "================================================================================"
    echo
    echo -e "\033[1;32m🌟 $2 🌟\033[0m"  # Green color for the project name
    echo
    sleep 1
}

# Call the logo display function
show_logo "WELCOME TO THE PROJECT SETUP TOOL" "Setting up your environment for the Bite Sized Project!"

# Function to check if Conda is installed
check_conda_installed() {
    if ! command -v conda &> /dev/null; then
        return 1
    else
        return 0
    fi
}

# Step 1: Determine the operating system
is_mac_silicon=false
if [[ $(uname -s) == "Darwin" && $(uname -m) == "arm64" ]]; then
    is_mac_silicon=true
fi

# Step 2: Install Conda if not installed
if ! check_conda_installed; then
    echo "Conda is not installed. Installing now..."
    if $is_mac_silicon; then
        echo "Detected Apple Mac Silicon"
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/.miniconda3
        rm Miniconda3-latest-MacOSX-arm64.sh
    else
        echo "Detected Linux"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/.miniconda3
        rm Miniconda3-latest-Linux-x86_64.sh
    fi

    # Export Conda path
    export PATH="$HOME/.miniconda3/bin:$PATH"
    echo "export PATH=\"$HOME/.miniconda3/bin:\$PATH\"" >> ~/.bashrc
    source ~/.bashrc

    # Initialize Conda for the shell
    conda init
    source ~/.bashrc
else
    echo "Conda is already installed."
    export PATH="$HOME/.miniconda3/bin:$PATH" # Ensure PATH is set
    conda init
    source ~/.bashrc
fi

# Step 3: Provide options for moving to the project folder
echo "Select a project folder to navigate to:"
echo "1. text_classification"
echo "2. etc"
read -p "Enter the number corresponding to your choice: " choice

case $choice in
    1)
        project_folder="text_classification"
        ;;
    2)
        project_folder="etc"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Navigate to the selected project folder
if [ -d "$project_folder" ]; then
    cd "$project_folder"
    echo "Moved to folder: $project_folder"
else
    echo "Folder $project_folder does not exist. Exiting."
    exit 1
fi

# Step 4: Check if Conda environment exists, create if it doesn't
if conda env list | grep -q "$project_folder"; then
    echo "Conda environment $project_folder already exists. Activating..."
else
    echo "Conda environment $project_folder does not exist. Creating with Python 3.10..."
    conda create -n $project_folder python=3.10 -y
fi

show_logo "SETTING COMPLETE!" "RUN [conda activate $project_folder] TO START!"