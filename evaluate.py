import argparse
from pathlib import Path
import logging
import glob
import re
import tarfile

from espnet2.bin.asr_inference import inference, get_parser
import subprocess


class FILE_IO:
    @staticmethod
    def replace(before: str, after: str, filepath: str):
        """
        replace any text in filepath
        """
        file_path = Path(filepath)
        assert file_path.exists(), f"file:{filepath} is not exist. so cannot replace"
        content = file_path.read_text()
        content = content.replace(before, after)
        file_path.write_text(content)

    @staticmethod
    def get_last_model_path(exp_dir: str) -> str:
        """
        Get the last model path in exp/
        Args:
            exp_dir(str): Directory path where the model file is located
        Return:
            str: last epoch model path string
        """
        max_num = -1
        model_path = ""
        for model in glob.glob(f"{exp_dir}/*epoch.pth"):
            epoch_num = int(re.sub(r"\D*(\d*)epoch.pth", r"\1", model))
            if max_num < epoch_num:
                max_num = epoch_num
                model_path = model
        return model_path

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

    def unzip_model(model_dir: Path, dist_dir: Path) -> Path:
        """
        unzip model.tar.gz
        Args:
            model_dir(str): directory that have model.tar.gz
            dist_dir(str): Distination to unzip
        Return:
            Path
        """

        tar_gz_file = model_dir / "model.tar.gz"
        assert tar_gz_file.exists(), "model.tar.gz file doesn't exist!!"

        with tarfile.open(tar_gz_file) as tar:
            tar.extractall(str(dist_dir))
            parent = list(tar)[0].name
        return dist_dir / parent


def add_parser(parser: argparse.ArgumentParser):
    # for decoder
    parser.add_argument("--asr_model_dir", type=str, help="Directory path containing ASR model.tar.gz")
    parser.add_argument(
        "--lm_dir",
        type=str,
        help="Directory path containing LM model.tar.gz",
    )

    # for scoerrer
    parser.add_argument("--reference_dir", type=str, help="Directory that have reference text")
    parser.add_argument("--score_output", type=str, help="output directory path for score ")
    return parser


def main(cmd=None):
    parser = get_parser()
    parser = add_parser(parser)
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    # fix wav path
    for i, (path, name, type) in enumerate(args.data_path_and_name_and_type):

        scp_path = Path(path)
        output_dir = Path(scp_path.parent.name)

        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "wav.scp"

        content = scp_path.read_text()
        content = content.replace("dump/raw/", "/opt/ml/processing/input/")
        output_file.write_text(content)
        args.data_path_and_name_and_type[i] = (str(output_file), name, type)

    print(args.data_path_and_name_and_type)
    # unzip model.tar.gz
    distination = Path("/opt/ml/model")
    # # ASR
    asr_model_dir = FILE_IO.unzip_model(Path(args.asr_model_dir), distination / "asr_model")
    args.asr_train_config = str(asr_model_dir / args.asr_train_config)
    FILE_IO.replace("/opt/ml/input/data/", "/opt/ml/processing/input/", args.asr_train_config)
    args.asr_model_file = FILE_IO.get_last_model_path(asr_model_dir)
    logging.info(f"The name of the ASR model used for evaluation is {args.asr_model_file}.")

    # # LM
    lm_dir = FILE_IO.unzip_model(Path(args.lm_dir), distination / "language_model")
    args.lm_train_config = str(lm_dir / args.lm_train_config)
    FILE_IO.replace("/opt/ml/input/data/", "/opt/ml/processing/input/", args.lm_train_config)
    args.lm_file = FILE_IO.get_last_model_path(lm_dir)
    logging.info(f"The name of the ASR model used for evaluation is {args.lm_file}.")

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
        proc = subprocess.Popen(cmd, encoding="utf-8", stdout=wf, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    retcode = proc.returncode

    if retcode == 1:
        raise RuntimeError(f"Occored error with below error \n{stderr}")
    del proc


if __name__ == "__main__":
    main()
