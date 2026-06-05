# LLM Sense and Sensibility

Presentation to de-mystify LLMs

## Running the demos on your machine

**Prerequisite:** [uv is installed](https://docs.astral.sh/uv/getting-started/installation/). `uv` is a fast, popular Python package manager that can run
Python applications on your local machine with no setup.

Run the demos:

1. Open the terminal on your machine. On Windows, this is the command prompt
   (cmd.exe), PowerShell, or the Windows Terminal. On a Macbook, you can use
   the Terminal application (Terminal.app)
2. Enter the following command into your terminal to run the demos:

    ```bash
    uvx --from git+https://github.com/bsweger/llm-sense-and-sensibility llmdemo
    ```

## Deploying the demos to Streamlit Community Cloud

The demos can be published as a public app on
[Streamlit Community Cloud](https://streamlit.io/cloud).

1. Push this repository to GitHub (it must be public).
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app**
   and point it at this repository.
3. Set the **Main file path** to `src/llm_sas/demos/demos.py`.
4. Under **Advanced settings**, set the **Python version** to `3.13`.
5. Deploy.

Notes:

- Streamlit Community Cloud installs from `src/llm_sas/demos/requirements.txt`,
  which is exported from `uv.lock` by a
  [pre-commit hook](scripts/gen_cloud_requirements.sh). On Linux it pins the
  CPU-only `torch` build, so the multi-GB CUDA stack is never downloaded.
- Every model in the demo is public on the Hugging Face Hub (no secrets needed).
- Streamlit Community Cloud has a 1 GB memory limitation. Thus, only one model
  at a time is cached.

## Project setup (for local development)

If you want to modify the code in this project (_e.g._, change the demos
or slides), see [CONTRIBUTING.md](CONTRIBUTING.md) for a guide to setting up
the project on your local machine.

## License

The content of this project itself is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/),
and the underlying source code used to calculate, format and display that
content is licensed under GNU General Public License v3.0.
