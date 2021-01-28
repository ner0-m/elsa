from log import get_logger
import gitlab
import argparse
import sys
import logging


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=
        "Creates a release for a project with the given name tag and summary (given as markdown file)"
    )
    parser.add_argument(
        "--gitlab-url",
        default="https://gitlab.lrz.de/",
        help="Gitlab instance url (default: \'https://gitlab.lrz.de/\')")
    parser.add_argument(
        "--project-id",
        default=30774,
        help=
        "Markdown which will be used as a summary (default: \'30774\' elsa)")
    parser.add_argument("--private-token",
                        required=True,
                        help="The private token to connect to GitLab instance")
    parser.add_argument(
        "--ref",
        default="master",
        required=False,
        help="Branch for which release should be created (default \'master\')")
    parser.add_argument("--name",
                        default="",
                        required=True,
                        help="Release name")
    parser.add_argument("--tag", default="", required=True, help="Release tag")
    parser.add_argument(
        "--summary-file",
        default="summary.md",
        help=
        "A markdown file, which will be used a a summary of the release (default \'summary.md\')"
    )
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Increase verbosity of the script")
    cl_args = parser.parse_args(argv)
    return vars(cl_args)


def create_release(gitlab_url, project_id, private_token, ref, name, tag,
                   summary_file, verbose):
    if verbose:
        logger = get_logger('Release', level=logging.DEBUG)
    else:
        logger = get_logger('Release', level=logging.INFO)

    logger.debug(f"GitLab instance: {gitlab_url}")
    logger.debug(f"Connection with token: {private_token}")
    logger.info(f"Connecting to GitLab")

    # Connect and authenticate to the GitLab instance
    gl = gitlab.Gitlab(gitlab_url, private_token=private_token)
    gl.auth()

    #Get project from GitLab
    project = gl.projects.get(project_id)

    logger.debug(f"Project name {project.name}")
    logger.debug(f"Project id {project_id}")

    with open(summary_file, 'r') as file:
        summary = file.read()

    logger.debug(f"Summary: \n-----------------\n{summary}\n-----------------")

    try:
        release = project.releases.create({
            'name': name,
            'tag_name': tag,
            'description': summary,
            'ref': ref
        })
    except AttributeError as e:
        logger.critical(
            f"Could not create release --> {e}\nYou are missing some information"
        )
    except gitlab.exceptions.GitlabCreateError as e:
        logger.critical(f"Could not create release --> {e}")


if __name__ == '__main__':
    parsed_arguments = parse_arguments(sys.argv[1:])
    create_release(**parsed_arguments)
