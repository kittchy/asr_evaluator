import argparse
from pathlib import Path
import logging

from espnet2.bin.asr_inference import inference, get_parser
import subprocess


class FILE_IO:
    def make_trn_file(text_file_path: Path, dist_path: Path, basename: str):
        """
        make text file as <basename>.trn
        """
        logging.info(f"make {basename}.trn file")
        if not dist_path.exists():
            dist_path.mkdir()

        dist_file = dist_path / (basename + ".trn")

        # read trn file
        with open(text_file_path) as rf:
            lines = rf.readlines()
        new_lines = []
        for line in lines:
            id_text = line.rstrip().split(" ")
            id = id_text[0]
            tokens = " ".join(list(id_text[1]))
            new_lines.append(f"{tokens}\t ({id})\n")

        # write trn file
        with open(dist_file, "w") as wf:
            wf.writelines(new_lines)

        return dist_file


def add_parser(parser: argparse.ArgumentParser):

    # for scoerrer
    parser.add_argument(
        "--reference_dir", type=str, help="Directory that have reference text"
    )
    parser.add_argument(
        "--score_output", type=str, help="output directory path for score "
    )
    return parser


def main(cmd=None):
    parser = get_parser()
    parser = add_parser(parser)
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    # inference
    inference(**kwargs)

    # scoring CER
    inference_text = Path(args.output_dir) / "1best_recog" / "text"
    reference_text = Path(args.reference_dir) / "text"
    distpath = Path(args.score_output)

    # create inf.trn, ref.trn
    inf_file = FILE_IO.make_trn_file(inference_text, distpath, "hyp")
    ref_file = FILE_IO.make_trn_file(reference_text, distpath, "ref")
    output_score_file = Path(args.score_output) / "result.txt"

    # calc score with sclite

    cmd = [
        "sclite",
        "-r",
        f"{ref_file}",
        "trn",
        "-h",
        f"{inf_file}",
        "-i",
        "rm",
        "-o",
        "all",
        "stdout",
    ]
    logging.info(" ".join(cmd))

    with open(output_score_file, "w") as wf:
        proc = subprocess.Popen(
            cmd, encoding="utf-8", stdout=wf, stderr=subprocess.PIPE
        )
    _, stderr = proc.communicate()
    retcode = proc.returncode

    if retcode == 1:
        raise RuntimeError(f"Occored error with below error \n{stderr}")
    del proc


if __name__ == "__main__":
    main()
