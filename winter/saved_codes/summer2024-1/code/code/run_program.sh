# Convert the C++ source file to Unix format (optional, if needed)
#sed -i 's/\r$//' program.cpp

# Compile the C++ program
#g++ -o ./bin/gdMain program.cpp

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program with arguments from 0 to 100..."
    # Run the compiled program with arguments from 0 to 100
    for i in {0..199}
    do
        echo "Running with argument $i..."
        ./bin/gdMain $i
    done
else
    echo "Compilation failed. Please check your code for errors."
fi