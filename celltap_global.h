#pragma once

#ifndef CELL_TAP_GLOBAL_H
#define CELL_TAP_GLOBAL_H 1

#include "celltap_namespace.h"

#ifdef CELLTAP_EXPORTS
#define CELLTAP_API __declspec(dllexport)
#else
#define CELLTAP_API __declspec(dllimport)
#endif

#endif //CELL_TAP_GLOBAL_H