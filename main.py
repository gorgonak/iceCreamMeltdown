from tkinter import *
from data import *
from PIL import Image, ImageTk
import time

root = Tk()
root.title('Ice Cream Meltdown')

root.geometry("480x250")

X, Y = import_training_data()
print("\n\t *** Training *** \n")
w = training(X, Y)
print("\n\t *** Training Completed ***\n")

title_label = Label(root, text='*** Ice Cream Meltdown ***', font=('calibre', 20, 'bold'), pady=10)
greet = Text(root, height=3, width=40, wrap=WORD)
greet.insert(INSERT, 'I will help you predict the amount of ice cream cones you will sell '
                     'based off the temperature outside.')

temp_var = StringVar()


def submit():
    temp = int(temp_var.get())
    print("The temperature = ", temp)
    temp_var.set("")

    outside_temp = temp
    prediction(outside_temp, w)

    cone_text = str("%d" % round(predict(float(outside_temp), w)))
    temp_text = str(outside_temp)

    result_text = "Based on " + temp_text + " degree temperatures,\nYou should sell ~" + cone_text + " ice cream cones."

    result = Text(root, height=2, width=40, wrap=WORD)
    result.insert(INSERT, result_text)
    result.grid(row=5, column=1)

    clean_up()


temp_label = Label(root, text='Temp', font=('calibre', 10, 'bold'), padx=10)
temp_entry = Entry(root, width=45, textvariable=temp_var, font=('calibre', 10, 'bold'))

sub_button = Button(root, text='Submit', command=submit)  # notice the command is our method from earlier
graph_button = Button(root, text='Show Graph', command=lambda: plot_chart(X, Y, w))

title_label.grid(row=0, column=1)
greet.grid(row=2, column=1)

temp_label.grid(row=3, column=0)
temp_entry.grid(row=3, column=1)
sub_button.grid(row=3, column=2)
graph_button.grid(row=7, column=1)

root.grid_rowconfigure(1, minsize=15)
root.grid_rowconfigure(4, minsize=15)
root.grid_rowconfigure(6, minsize=15)

root.mainloop()
