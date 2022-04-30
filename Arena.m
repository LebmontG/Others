classdef Policy < handle
    properties
        p;
        start;
        init;
        map;
        present;
        dif;
        int;
        param;
        prever;
%        properties
%         open_list;
%         pre_loc;
%         pre;
%         last;
%         last0;
%         alpha;
%         weights;
%         weights0;
%         hidden_layer;
%         inp_dim;
%         out_dim;
%         memory_size;
%         state_buffer;
%         next_state_buffer;
%         reward_buffer;
%         action_buffer;
%         action_buffer0;
%         batchsize;
%         buf_count;
%         action_space;
%         action_space0;
%         gamma;
%         map;
    end

    methods
        function self = Policy()
            self.prever=0;
            self.param=[1;0;0.4];
            self.init=0;
            self.int=0;
            self.dif=0;
            self.map=zeros(50,50);
%            self.open_list=[];
%             self.map=zeros(50,50);
%             self.pre_loc=[0,0];
%             self.pre=0;
%             self.inp_dim=3;
%             self.out_dim=3;
%             self.action_space=[-0.2,0,0.2];
%             self.action_space0=[-0.5,10,0.5];
%             self.gamma=0.8;
%             self.batchsize=12;
%             self.buf_count=0;
%             self.memory_size =120;
%             self.state_buffer=zeros(self.memory_size,self.inp_dim);
%             self.next_state_buffer=zeros(self.memory_size,self.inp_dim);
%             self.action_buffer=zeros(1,self.memory_size);
%             self.action_buffer0=zeros(1,self.memory_size);
%             self.reward_buffer=zeros(1,self.memory_size);
%             %self.buffer(1:self.memory_size)=struct('state',[],'action',[],'next_state',[],'reward',[],'done',[]);
%             self.alpha =0.003;
%             self.hidden_layer =[8,8];
%             hidden_layer=self.hidden_layer;
%             self.weights.input = normrnd(0,1,[1+self.inp_dim,hidden_layer(1)])/sqrt(self.inp_dim);
%             self.weights.hidden = normrnd(0,1,[hidden_layer(1)+1, hidden_layer(2)])/sqrt(hidden_layer(1));
% 			self.weights.out = normrnd(0,1,[hidden_layer(2)+1,self.out_dim])/sqrt(hidden_layer(2));
%             self.weights0.input = normrnd(0,1,[1+self.inp_dim,hidden_layer(1)])/sqrt(self.inp_dim);
%             self.weights0.hidden = normrnd(0,1,[hidden_layer(1)+1, hidden_layer(2)])/sqrt(hidden_layer(1));
% 			self.weights0.out = normrnd(0,1,[hidden_layer(2)+1,self.out_dim])/sqrt(hidden_layer(2));
        end
        
        function action=action(self,observation)
            if self.init==0
                self.initiate(observation);
                self.init=1;
            end
            if observation.collide==1
                action=[-10,0];
            else
                x0=observation.agent.x;
                y0=observation.agent.y;
                l=length(self.p);
                ind=2;
                dis=realmax;
                for i=1:l
                    d=sqrt((self.p(i).x-x0)^2+(self.p(i).y-y0)^2);
                    if dis>d
                        ind=i;
                        dis=d;
                    end
                end
                if ind>1
                    ind=ind-1;
                end
%                 loc=self.update(observation);
%                 if loc~=-1
%                     x0=round(observation.agent.x);
%                     y0=round(observation.agent.y);
%                     path=self.DAstar([self.p(loc).x,self.p(loc).y],[x0,y0]);
%                     self.p=[self.p(1:loc),path];
%                 else
%                     x0=round(observation.agent.x);
%                     y0=round(observation.agent.y);
%                     dis=self.cost([self.p(1).x,self.p(1).y],[x0,y0]);
%                     loc=1;
%                     for i=1:length(self.p)
%                         if dis>sqrt((x0-self.p(i).x)^2+(y0-self.p(i).y)^2)
%                             loc=i;
%                             dis=sqrt((x0-self.p(i).x)^2+(y0-self.p(i).y)^2);
%                         end
%                     end
%                     path=self.DAstar([self.p(loc).x,self.p(loc).y],[x0,y0]);
%                     self.p=self.p(1:loc);
%                 end
                action=self.move(observation,self.p(ind));
            end
        end

        function action=move(self,observation,target)
            x0=observation.agent.x;
            y0=observation.agent.y;
            h0=mod(observation.agent.h,2*pi);
            x0=x0+0.4*cos(h0);
            y0=y0+0.4*sin(h0);
            difx=x0-target.x;
            dify=y0-target.y;
            er=(sin(h0-atan2(dify,difx)))*sqrt(difx^2+dify^2);
            self.int=self.int+er;
            self.dif=er-self.dif;
            an=[er,self.int,self.dif]*self.param;
            self.dif=er;
            self.prever=er;
            action=[10,an];
%             curp(1) = curp(1) + dt*v*cos(curp(3));
%             curp(2) = curp(2) + dt*v*sin(curp(3));  
%             curp(3) = curp(3) + dt*v*tan(u)/L;
%             l=length(self.p)-2;
%             target=[self.p(l).x,self.p(l).y];
%             x0=observation.agent.x;
%             y0=observation.agent.y;
%             disp(target);
%             disp([x0,y0]);
%             h0=mod(observation.agent.h,2*pi);
%             bias_an=atan((target(2)-y0)/(target(1)-x0))-h0;
%             if (target(1)-x0)<0
%                 bias_an=bias_an+pi;
%             end
%             self.dif=self.dif+bias_an;
%             self.int=self.int+bias_an;
%             an=-[bias_an,self.int,self.dif]*self.param;
%             self.dif=bias_an;
%             action=[0.9,an];
            %bias_dis=(sqrt((target(2)-y0)^2+(target(1)-x0)^2));
%             if abs(bias_an)<0.1
%                 action=[0.9,0];
%             else
%                 action=[0,bias_an];
%             end
        end
        
        function initiate(self,observation)
            self.map=observation.map;
            self.complement();
            update(self);
            x0=round(observation.agent.x);
            y0=round(observation.agent.y);
            self.start=[observation.endPos.x,observation.endPos.y];
            path=self.DAstar(self.start,[x0,y0]);
            self.p=path;
            self.present=length(path);
%             for i=1:self.present-1
%                 x0=self.p(i).x;
%                 y0=self.p(i).y;
%                 xn=self.p(i+1).x;
%                 yn=self.p(i+1).y;
%                 if abs(x0-xn)==1 && 1==abs(y0-yn)
%                     if self.map(x0,yn)>0 && self.map(xn,y0)==0
%                         n=struct('x',xn,'y',y0,'g',self.p(i+1).g,'h',self.p(i+1).h);
%                         self.p=[self.p(1:i),n,self.p(i+1:self.present)];
%                     elseif self.map(x0,yn)==0 && self.map(xn,y0)>0
%                         n=struct('x',x0,'y',yn,'g',self.p(i+1).g,'h',self.p(i+1).h);
%                         self.p=[self.p(1:i),n,self.p(i+1:self.present)];
%                     end
%                 end
%             end
        end
 
        function neighb=get_nei(~,x,y)
        neighb=[];
        for i=-1:1
            for j=-1:1
                x0=x+i;
                y0=y+j;
                if x0<1||x0>50||y0<1||y0>50||(i==0&&j==0)
                    continue;
                end
%                 if j==0
%                     continue;
%                 end
                s=struct('x',x0,'y',y0,'g',0,'h',0);
                neighb=[neighb,s];
            end
        end
        end
     
        function close=DAstar(self,begin,dest)
            %[row,col]=size(self.map);
            open=struct('x',-1,'y',-1,'g',-1,'h',-1);
            openlen=0;
            endx=dest(1);
            endy=dest(2);
            startx=begin(1);
            starty=begin(2);
            close=struct('x',startx,'y',starty,'g',0,'h',0);
            closelen=1;
            for i=self.get_nei(close(1).x,close(1).y)
                if self.map(i.x,i.y)~=1
                    openlen=openlen+1;
                    open(openlen).x=i.x;
                    open(openlen).y=i.y;
                    open(openlen).g=1;
                    open(openlen).h=self.cost([i.x,i.y],[endx,endy]);
                end
            end

            while openlen>0
                min = realmax;
                for i=1:openlen
                    if open(i).g+open(i).h<=min
                        min=open(i).g+open(i).h;
                        ind=i;
                    end
                end
                close(closelen+1).x=open(ind).x;
                close(closelen+1).y=open(ind).y;
                close(closelen+1).g=open(ind).g;
                close(closelen+1).h=open(ind).h;
                closelen=closelen+1;
                openlen=0;
                if close(closelen).x == endx && close(closelen).y==endy
                    break;
                end
                for i=self.get_nei(close(closelen).x,close(closelen).y)
                    if self.map(i.x,i.y)==0
                       flag=false;
                       for m=1:closelen
                           if close(m).x==i.x && close(m).y==i.y
                               flag=true;
                               break;
                           end
                       end
                       if flag
                            continue;
                       end
                       flag=false;
                       for m=1:openlen
                           if open(m).x==i.x && open(m).y==i.y
                               flag=true;
                               break;
                           end
                       end
                       if flag
                            continue;
                       else
                            open(openlen+1).x=i.x;
                            open(openlen+1).y=i.y;
                            openlen=openlen+1;
                            open(openlen).g=close(closelen).g+1;
                            open(openlen).h=self.cost([endx,endy],[open(openlen).x,open(openlen).y]);
                       end
                    end
                end
            end
            flag=false;
            for i=1:closelen
                if close(i).x==endx && close(i).y==endy
                    flag=true;
                    break;
                end
            end
            if flag==0
                for i=1:closelen
                    self.map(close(i).x,close(i).y)=2;
                end
                close=self.DAstar(begin,dest);
            end
        end
            
        function dis=cost(~,x,y)
            dis=abs(x(1)-y(1))+abs(x(2)-y(2));
        end
        
        function loc=update(self)
            loc=-1;
            for i=1:50
                for j=1:50
                    if self.map(i,j)==1
                        self.expand(i,j);
                    end
                end
            end
            %self.map=zeros(50,50);
%             for i=1:length(self.p)
%                 if self.map(self.p(i).x,self.p(i).y)>0
%                     loc=i;
%                     break;
%                 end
%             end
        end
        
        function expand(self,x,y)
            for i=-1:1
                for j=-1:1
                    if i+j==2
                        c=1;
                    elseif abs(j)+abs(i)==2
                        continue;
                    end
                    x0=x+i;
                    y0=y+j;
                    if y0<51 && y0>0 && x0>0 && x0<51
                        if self.map(x0,y0)==0
                            self.map(x0,y0)=2;
                        end
                    end
                end
            end
        end
        
        function complement(self)
            for i=1:50
                for j=1:50
                    if self.map(i,j)==1
                        if i>1
                            self.map(i-1,j)=1;
                        end
                        if j>1
                            self.map(i,j-1)=1;
                        end
                        if i>1&&j>1
                            self.map(i-1,j-1)=1;
                        end
                        %见了鬼了为啥同样的逻辑结果不一样
                        %传统美德matlab
%                         for loc=[[-1,-1];[-1,0];[0,-1];[0,0]]
%                             x=i+loc(1);
%                             y=j+loc(2);
%                             if y<51 && y>0 && x>0 && x<51
%                                 self.map(x,y)=1;
%                             end
%                         end
                    end
                end
            end
        end
        
        function value=forward(self,features)
            value.hidden_in_value = [1 features] * self.weights.input;
            value.hidden_out_value = sigmoid(value.hidden_in_value);
            value.hidden_in_value2 = [1 value.hidden_out_value] * self.weights.hidden;
            value.hidden_out_value2 = sigmoid(value.hidden_in_value2);
            value.out_value = [1 value.hidden_out_value2] * self.weights.out;
        end
        
        function value=forward0(self,features)
            value.hidden_in_value = [1 features] * self.weights0.input;
            value.hidden_out_value = sigmoid(value.hidden_in_value);
            value.hidden_in_value2 = [1 value.hidden_out_value] * self.weights0.hidden;
            value.hidden_out_value2 = sigmoid(value.hidden_in_value2);
            value.out_value = [1 value.hidden_out_value2] * self.weights0.out;
        end
        
        function state=get_state(self,observation)
            dest=[observation.endPos.x,observation.endPos.y];
            x0=observation.agent.x;
            y0=observation.agent.y;
            h0=mod(observation.agent.h+pi,2*pi);
            %state=[self.get_obs(observation),dest(1)-x0,dest(2)-y0,h0];
            state=[dest(1)-x0,dest(2)-y0,h0];
            
%             dest=[observation.endPos.x,observation.endPos.y];
%             an=mod(observation.agent.h+pi,2*pi);
%             col=observation.collide;
%             x0=observation.agent.x;
%             y0=observation.agent.y;
%             loc=1+round(an/(pi/4));
%             ran=[[-8,-4];[-8,-8];[-4,-8];[0,-8];[0,-4];[0,0];[-4,0];[-8,0];[-8,-4]];
%             x=round(x0+ran(loc,1));
%             y=round(y0+ran(loc,2));
%             x=max(min(x,43),1);
%             y=min(max(y,1),43);
%             obs=observation.scanMap(x:x+7,y:y+7);
%             obs=reshape(obs,1,[]);
%             state=[dest(1)-x0,dest(2)-y0,an,obs];
        end
        
        function reward=get_rew(self,state)
            %reward sparse!
            x=state(1);
            y=state(2);
            if  x>49||x<1||y>49||y<1
                reward=-1;
            else
                %reward=1;
                dis=sqrt(x^2+y^2);
                if dis>=sqrt(self.pre_loc(1)^2+self.pre_loc(2)^2)
                    reward=-1;
                else
                    reward=1;
                end
                self.pre_loc(1)=x;
                self.pre_loc(2)=y;
                %reward=1-state(3)-sqrt(state(1)^2+state(2)^2)/70;
            end
        end
        
        function obs=get_obs(~,observation)
            x0=observation.agent.x;
            y0=observation.agent.y;
            radius=2;
            h0=mod(observation.agent.h+pi,2*pi);
            head=[x0+3*cos(h0),y0+3*sin(h0)];
            h1=[head(1)+radius*cos(h0+pi/2),head(2)+radius*sin(h0+pi/2)];
            h2=[head(1)+radius*cos(h0-pi/2),head(2)+radius*sin(h0-pi/2)];
            tail=[x0-3*cos(h0),y0-3*sin(h0)];
            t1=[tail(1)+radius*cos(h0+pi/2),tail(2)+radius*sin(h0+pi/2)];
            t2=[tail(1)+radius*cos(h0-pi/2),tail(2)+radius*sin(h0-pi/2)];
            loc=round([head;h1;h2;tail;t1;t2]);
            obs=ones(1,6);
            l=length(loc);
            for i=1:l
                obs(i)=restrict(observation.scanMap,loc(i,1),loc(i,2));
            end
        end
        
        function target=sch(self,observation)
            for i=1:50
                for j=1:50
                    if observation.scanMap(i,j)==1
                        self.map(i,j)=1;
                    end
                end
            end
            dest=[observation.endPos.x,observation.endPos.y];
            x0=round(observation.agent.x);
            y0=round(observation.agent.y);
            st=self.allpoint(x0,y0);
            de=self.allpoint(dest(1),dest(2));
            tm=st;
            while tm~=de
                if self.map(tm.x,tm.y)==1
                    self.modify(tm);
                    continue;
                end
                tm=tm.par;
            end
            target=st.par;
        end
        
        function kmin=process_state(self)
            if length(self.open_list)~=0
                m=self.open_list(1);
                for x=self.open_list
                    if x.k<m.k
                        m=x;
                    end
                end
                k_pre=m.k;
                self.remove(m);
                if k_pre<m.h
                    n=self.get_nei(x.x,x.y);
                    for y=n
                        if (y.h<k_pre)&&(x.h>y.h+cost(x,y))
                            x.par=y;
                            x.h=y.h+cost(x,y);
                        end
                    end
                elseif k_pre==m.h
                    n=self.get_nei(x.x,x.y);
                    for y=n
                        if y.flag== "new" ||((y.par == x)&&(y.h~=x.h +cost(x,y)))||((y.par~=x)&&(y.h > x.h +cost(x,y)))
                            y.par=x;
                            self.insert(y,x.h+cost(x,y));
                        end
                    end
                else
                    n=self.get_nei(x.x,x.y);
                    for y=n
                        if y.flag=="new"||((y.par==x)&&(y.h~=x.h+cost(x,y)))
                            y.par=x;
                            self.insert(y,x.h+cost(x,y));
                        else
                            if y.par~=x && y.h>x.h+cost(x,y)
                                self.insert(x,x.h);
                            else
                                if y.par~=x && x.h>y.h+cost(x,y)&&y.flag=="close" &&y.h>k_pre
                                    self.insert(y,y.h);
                                end
                            end
                        end
                    end
                end
            end
            kmin=self.kmin();
            s=self.allpoint(self.startx,self.starty);
            if s.flag=="close"
                self.start=1;
            end
        end
        
        function k=kmin(self)
            if length(self.open_list)==0
                k=-1;
            else
                k=self.open_list(1);
                k=k.k;
                for i=self.open_list
                    k=min(i.k,k);
                end
            end
        end
        
        function none=modify(self,state)
            if state.flag=="close"
                self.insert(state, state.par.h +state.cost(state.par));
            end
            while 1
                km=self.prosess_cost();
                if km>=state.h
                    break;
                end
            end
        end
        
        function remove(self,state)
            if state.flag=="open"
                state.flag="close";
            end
            self.del(state);
        end
        
        function del(self,state)
            if length(self.open_list)~=0
                for i=1:length(self.open_list)
                    if (self.open_list(i).x==state.x)&&(self.open_list(i).y==state.y)
                        self.open_list(i)=[];
                        break;
                    end
                end
            end
        end
        
        function none=insert(self,state,h_new)
            loc=[state.x,state.y];
            e=0;
            for x=self.open_list
                if x.x==loc(1)&&x.y==loc(2)
                    e=1;
                    break;
                end
            end
            if e==0
                if state.flag== "new"
                    state.k = h_new;
                elseif state.flag == "open"
                    state.k = min(state.k, h_new);
                elseif state.flag == "close"
                    state.k = min(state.h, h_new);
                end
                state.h = h_new;
                state.flag = "open";
                self.open_list=[state,self.open_list];
            end
        end
        
    end
end

% classdef Policy < handle
%     properties
%         gamma
%         clip
%         epochs
%         std
%         std_decay
%         min_std
%         action_dim
%         state_dim
%         action_range
%         buf_act
%         buf_sta
%         buf_log
%         buf_rew
%         buf_ter
%         var
%         actor
%         critic
%         t
%     end
% 
%     methods
%         function self = Policy()
%             self.t=0;
%             self.action_range=[-1,1];
%             self.state_dim=[1,67];
%             self.action_dim=[1,1];
%             self.gamma=0.8;
%             self.std=0.6;
%             self.std_decay=0.0001;
%             self.min_std=0.02;
%             self.var=self.std*self.std;
%             self.clip=0.2;
%             self.epochs=1;
%             self.buf_act=[];
%             self.buf_sta=[];
%             self.buf_log=[];
%             self.buf_rew=[];
%             self.buf_ter=[];
%             inp=50*ones(67,1);
%             inp(1)=-50;
%             self.actor=newff(minmax(inp),[32,8,1],{'tansig' , 'tansig','purelin'},'traingd');
%             self.critic=newff(minmax(inp),[32,8,1],{'tansig' , 'tansig','purelin'},'traingd');
%             self.actor=init(self.actor);
%             self.critic=init(self.critic);
%             self.critic.trainParam.epochs=10;
%             self.actor.trainParam.epochs=10;
% %             self.actor=[imageInputLayer([50,50,1])
% %                 convolution2dLayer(3,1)
% %                 reluLayer
% %                 fullyConnectedLayer(10)
% %                 reluLayer
% %                 fullyConnectedLayer(2)];
% %             self.critic=[imageInputLayer([50,50,1])
% %                 convolution2dLayer(3,1)
% %                 reluLayer
% %                 fullyConnectedLayer(10)
% %                 reluLayer
% %                 fullyConnectedLayer(1)];
%         end
%         
%         function action=action(self,observation)
%             state=self.get_state(observation);
%             x=state(1);
%             y=state(2);
%             if state(3)
%                 action=[-10,0];
%             elseif x>48||x<2||y>48||y<2
%                 action=[-10,0];
%             else
%                 disp('?');
%                 m_a=sim(self.actor,state);
%                 disp(m_a);
%                 action=mvnrnd(m_a,self.var);
%                 prob=mvnpdf(action,m_a,self.var);
%                 self.buf_rew=[self.buf_rew,self.get_rew(state)];
%                 self.buf_act=[self.buf_act,action];
%                 self.buf_sta=[self.buf_sta,state];
%                 self.buf_log=[self.buf_log,log(prob)];
%                 self.t=self.t+1;
%                 if self.t==10
%                     self.update();
%                     self.t=0;
%                 end
%                 action=[10,action];
%             end
%         end
%         
%         function prob=evaluate(self,state,action)
%             m_a=sim(self.actor,state);
%             %action=mvnrnd(m_a,self.var);
%             prob=log(mvnpdf(action,m_a,self.var));
%             %value=sim(self.critic,state);
%         end
%         
%         function decay(self)
%             if self.std~=self.min_std
%                 self.std=self.std-seld.std_decay;
%             end
%         end
%         
%         function update(self)
%             %matlab too tough to code
%             self.buf_rew=flip(self.buf_rew);
%             %self.buf_ter=flip(self.buf_ter);
%             rewards=self.buf_rew;
%             for i=2:length(self.buf_rew)
%                 rewards(i)=self.gamma*rewards(i-1)+rewards(i);
%             end
%             rewards=zscore(flip(rewards));
%             for i=1:self.epochs
%                 l=length(self.buf_act);
%                 pro=ones(1,l);
%                 value=ones(1,l);
%                 v=sim(self.critic,self.buf_sta);
%                 for j=1:l
%                     p=self.evaluate(self.buf_sta(:,j),self.buf_act(j));
%                     pro(j)=exp(p-self.buf_log(j));
%                     value(j)=pro(j)*(rewards(j)-v(j));
%                 end
%                 %loss=self.gamma*sim(self.critic,self.buf_sta);
%                 %loss=(1-self.gamma)*[self.buf_rew(2:10),self.buf_rew(10)]+loss;
%                 loss=sim(self.actor,self.buf_sta);
%                 loss=loss+value;
%                 self.critic=train(self.critic,self.buf_sta,loss);
%                 self.actor=train(self.actor,self.buf_sta,loss);
%             end
%             self.buf_act=[];
%             self.buf_sta=[];
%             self.buf_log=[];
%             self.buf_rew=[];
%             %self.buf_ter=[];
%         end
%         
%         function state=get_state(~,observation)
%             dest=[observation.endPos.x,observation.endPos.y];
%             an=mod(observation.agent.h+pi,2*pi);
%             col=observation.collide;
%             x0=observation.agent.x;
%             y0=observation.agent.y;
%             loc=1+round(an/(pi/4));
%             ran=[[-8,-4];[-8,-8];[-4,-8];[0,-8];[0,-4];[0,0];[-4,0];[-8,0];[-8,-4]];
%             x=round(x0+ran(loc,1));
%             y=round(y0+ran(loc,2));
%             x=max(min(x,43),1);
%             y=min(max(y,1),43);
%             obs=observation.scanMap(x:x+7,y:y+7);
%             obs=reshape(obs,1,[]);
%             state=[dest(1)-x0,dest(2)-y0,col,obs]';
%         end
%         
%         function reward=get_rew(~,state)
%             reward=1-state(3)-sqrt(state(1)^2+state(2)^2)/100;
%         end
%     end
% end

% classdef Policy < handle
%     properties
%         agent
%     end
%     methods
%         
%         function self = Policy(actor,critic,opt)
%             self.agent=rlPPOAgent(actor,critic,opt);
%         end
%         
%         function action=action(self,observation)
%             if observation.collide
%                 action=[-10,rand(1)-0.5];
%             else
%                 action=[10,rand(1)-0.5];
%             end
%         end
%         
%     end
% end
%save '1.mat' agent;
%load '1.mat' agent;
function r=restrict(map,x,y)
if x<1||x>50||y<1||y>50
    r=1;
else
    r=map(x,y);
    %scatter(x,y);
end
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function dis=cost(x,y)
dis=sqrt((x.x-y.x)^2+(x.y-y.y)^2);
end