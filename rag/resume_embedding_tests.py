
from resume_embedding import Resume

if __name__ == '__main__':
    resume = Resume("~/Documents/my-resume.pdf")
    result = resume.query("我在最后一份工作干了几年？")
    print(result)
