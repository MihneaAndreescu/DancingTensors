#include <iostream>
#include <string>
#include "KillIf.h"

void killIf(bool value, std::string message) {
	if (value) {
		std::cout << message << "\n";
		exit(0);
	}
}

