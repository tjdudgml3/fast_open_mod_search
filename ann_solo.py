import logging
from typing import List, Union

from ann_solo import spectral_library
from ann_solo import writer
from ann_solo.config import config


def ann_solo(spectral_library_filename: str, query_filename: str,
             out_filename: str, **kwargs: Union[bool, float, int, str]) -> int:
    """
    Run ANN-SoLo with the specified search settings.

    Values for search settings that are not explicitly specified will be taken
    from the config file (if present) or take their default values.

    The identified PSMs will be stored in the given file.

    Parameters
    ----------
    spectral_library_filename : str
        The spectral library file name.
    query_filename : str
        The query spectra file name.
    out_filename : str
        The mzTab output file name.
    **kwargs : Union[bool, float, int, str]
        Additional search settings. Keys MUST match the command line
        arguments (excluding the '--' prefix;
        https://github.com/bittremieux/ANN-SoLo/wiki/Parameters). Values
        MUST be the argument values. Boolean flags can be toggled by
        specifying True or False (ex: no_gpu=True).

    Returns
    -------
    int
        The error code from running ANN-SoLo.
    """
    # Convert kwargs dictionary to list for main().
    # 'args' contains arguments with values.
    # 'flags' contains boolean flags to include
    args = sum([['--' + k, str(v)] for k, v in kwargs.items()
                if not isinstance(v, bool)], [])
    flags = ['--' + k for k, v in kwargs.items() if v and isinstance(v, bool)]

    # Explicitly set the search parameters when run from Python.
    error_code = main([spectral_library_filename, query_filename, out_filename,
                       *args, *flags])

    return error_code


def main(args: Union[str, List[str]] = None) -> int:
    # Initialize logging.
    logging.basicConfig(format='{asctime} [{levelname}/{processName}] '
                               '{module}.{funcName} : {message}',
                        style='{', level=logging.DEBUG)
#     logging.debug(args)
#     exit()
    # Load the configuration.
    config.parse(args)

    # Perform the search.
    spec_lib = spectral_library.SpectralLibrary(
        config.spectral_library_filename)
    candidate_list = []
#     identifications = spec_lib.search(config.query_filename)
    identifications, candidate_list = spec_lib.search(config.query_filename, candidate_list)
    query_file = config.query_filename.split("/")[-1]
    with open(f"01_23_method_4096_allow_shift_sequest_each_cand.txt", "w") as f:
#     with open(f"7_22_{query_file}.txt", "w") as f:
        for query, candidate,std in candidate_list:
            if not candidate:
                continue
            f.write(f"{query},{std}\t")
            for cand in candidate:
                f.write(f"{cand},")
            f.write("\n")
    writer.write_mztab(identifications, config.out_filename,
                       spec_lib._library_reader)
    spec_lib.shutdown()

    logging.shutdown()

    return 0


if __name__ == '__main__':
    # Use search parameters from sys.argv when run from CMD.
    main()
