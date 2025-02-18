#!/bin/bash

# Function to display a cool welcome logo
show_logo() {
    echo "================================================================================"
    echo "\033[1;32m🚀 $1 🚀\033[0m"  # Green color for the project name
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
    echo "\033[1;32m🌟 $2 🌟\033[0m"  # Green color for the project name
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
fi

# Step 3: Provide options for moving to the project folder
folders=($(find . -maxdepth 1 -type d ! -name "." ! -name ".*" | sed 's|./||'))

# Display folder options
echo "Select a project folder to navigate to:"
for i in "${!folders[@]}"; do
    echo "$(($i + 1)). ${folders[$i]}"
done
echo "$(($i + 2)). Create a new folder"

# Get user choice
read -p "Enter the number corresponding to your choice: " choice

# Check if the choice is valid
if [[ "$choice" -ge 1 && "$choice" -le "${#folders[@]}" ]]; then
    project_folder="${folders[$((choice - 1))]}"
elif [[ "$choice" -eq "$((${#folders[@]} + 1))" ]]; then
    read -p "Enter the name of the new folder: " project_folder 
    mkdir -p "$project_folder"
    echo "Folder '$new_folder' created."
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Navigate to the selected project folder
if [ -d "$project_folder" ]; then
    cd "$project_folder"
    echo "Moved to folder: $project_folder"
	echo "Setting up basic folder..."
	mkdir -p 'config'
	mkdir -p 'notebook'
	mkdir -p 'src'
	cp '../common/requirements.txt' './config/requirements.txt'
	
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
