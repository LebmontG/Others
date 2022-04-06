classdef Policy < handle
    properties
        nn
        pre_s
        a_space
        gamma
        alpha
%        pre_a
%         b
%         c
%         h
%         w
%         w_1
%         w_2
%         xita
    end

    methods
        
        function self = Policy()
            self.gamma=0.8;
            self.nn=newff([-20,20],[5,10],{'tansig' , 'purelin'},'traingd');
            self.nn=init(self.nn);
            self.pre_s=4;
            %discrete actions for deep Q-Learning
            self.a_space=[0:1:9];
%             self.pre_a=5;
            self.alpha=0.9;
%             self.xita=0.3;
%             self.b=0.5*ones(4,1);
%             self.c=[0,1,-1,2];
%             self.h=zeros(4,1);
%             self.w=rands(4,1);
%             self.w_1=self.w;
%             self.w_2=self.w_1;
        end
        
        function action=action(self,observation)
            state=[observation.targetHeight-observation.agent.H];
            reward=sign(abs(self.pre_s)-abs(state))-0.2*abs(state);
            self.nn.trainParam.epochs=500;
            pre_q=sim(self.nn,self.pre_s);
            [n,ind]=max(pre_q);
            q=reward+self.gamma*max(sim(self.nn,state));
            pre_q(ind)=self.alpha*q+(1-self.alpha)*pre_q(ind);
            self.nn=train(self.nn,self.pre_s,pre_q);
            q=sim(self.nn,state);
            [n,ind]=max(q);
            action=self.a_space(ind);
            self.pre_s=state;
%             state=[observation.targetHeight-observation.agent.H];
%             if self.height~=0
%                 reward=abs(self.height)-abs(state);
%                 reward=self.pre_a*sign(reward);
%                 self.update(reward);
%             end
%             self.height=state;
%             action=self.forward(state);
        end
        
%         function pred=forward(self,state)
%                 for j=1:1:4
%                     self.h(j)=exp(-norm(state-self.c(:,j))^2/(2*self.b(j)*self.b(j)));
%                 end
%                 pred=self.w'*self.h;
%         end
        
%         function none=update(self,reward)
%             d_w=-self.xita*(reward)*self.h;
%             self.w=self.w_1+ d_w+self.alpha*(self.w_1-self.w_2);
%             self.w_2=self.w_1;
%             self.w_1=self.w;
%             for j=1:1:4
%                 upd=self.alpha*reward*self.w(j)*self.h(j)/self.b(j)^3*norm(self.height-self.c(:,j))^2;
%                 self.b(j)=self.b(j)+upd;
%                 upd=reward*self.w(j)*self.h(j)/self.b(j)^2*self.height*self.c(j);
%                 self.c(j)=self.c(j)+upd;
%             end
%         end
    end
end
