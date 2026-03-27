
from resume_embedding import Resume

if __name__ == '__main__':
    resume = Resume("~/Documents/my-resume.pdf")
    result,docs = resume.query("我在美团工作了几年？")
    print(result)
